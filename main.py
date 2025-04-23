import os
import sys  
import time
import uuid
import json
import logging
import asyncio
import requests
from typing import Dict, Any, Optional, List, Union
from abc import ABC, abstractmethod
from captchaai import CaptchaAI
import uvicorn
from fastapi import FastAPI, HTTPException, Body, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel, Field

# Playwright
from playwright.async_api import async_playwright, Page, Browser, BrowserContext
from playwright.async_api import TimeoutError as PlaywrightTimeout, expect



# Optional Redis import - we'll handle missing package gracefully
try:
    import redis.asyncio as aioredis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False


if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())


# =======================
#   LOGGING
# =======================
logger = logging.getLogger("RecaptchaBrowser")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s [%(name)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)

# =======================
#   CONFIG
# =======================
# Storage type: "memory" or "redis"
STORAGE_TYPE = os.getenv("STORAGE_TYPE", "memory").lower()

# Redis config (only used if STORAGE_TYPE=redis)
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
API_KEY= os.getenv("API_KEY", 'b755df7ff3dea3991eaf74c10882cdfc')
# Browser and task settings
BROWSER_POOL_SIZE = int(os.getenv("BROWSER_POOL_SIZE", "3"))
TASK_EXPIRE_SECONDS = int(os.getenv("TASK_EXPIRE_SECONDS", "3600"))  # 1 hour
SOLVE_TIMEOUT = float(os.getenv("SOLVE_TIMEOUT", "90.0"))
CLEANUP_INTERVAL = int(os.getenv("CLEANUP_INTERVAL", "300"))  # 5 minutes
MAX_CONCURRENT_TASKS = int(os.getenv("MAX_CONCURRENT_TASKS", "10"))

# =======================
#   DATA MODELS
# =======================
class SubmitTaskPayload(BaseModel):
    captcha_id: str = Field(..., description="Recaptcha captcha ID")
    slot: int = Field(0, description="Optional slot or port for local test server")
    target_url: Optional[str] = Field(None, description="Optional URL to visit (overrides slot)")

class GetResultPayload(BaseModel):
    task_id: str = Field(...)

class TaskData(BaseModel):
    task_id: str
    status: str = "pending"
    error: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    created_at: float
    completed_at: Optional[float] = None
    captcha_id: str
    slot: int
    target_url: Optional[str] = None

# =======================
#   STORAGE INTERFACE
# =======================
class TaskStorage(ABC):
    """Abstract interface for task storage"""

    @abstractmethod
    async def initialize(self):
        """Initialize the storage"""
        pass

    @abstractmethod
    async def create_task(self, task_id: str, captcha_id: str, slot: int, target_url: Optional[str]) -> None:
        """Create a new task"""
        pass

    @abstractmethod
    async def get_task(self, task_id: str) -> Optional[TaskData]:
        """Get task data by ID"""
        pass

    @abstractmethod
    async def update_task(self, task_id: str, status: str, error: Optional[str] = None, 
                          result: Optional[Dict[str, Any]] = None) -> None:
        """Update task status"""
        pass

    @abstractmethod
    async def count_tasks_by_status(self) -> Dict[str, int]:
        """Count tasks by status"""
        pass

    @abstractmethod
    async def cleanup_old_tasks(self) -> int:
        """Clean up expired tasks, return count of cleaned tasks"""
        pass

    @abstractmethod
    async def shutdown(self) -> None:
        """Shutdown the storage"""
        pass

# =======================
#   MEMORY STORAGE
# =======================
class MemoryTaskStorage(TaskStorage):
    """In-memory implementation of task storage"""

    def __init__(self):
        self.tasks: Dict[str, TaskData] = {}
        self.lock = asyncio.Lock()

    async def initialize(self):
        logger.info("Initializing in-memory task storage")
        # Nothing to initialize for in-memory storage

    async def create_task(self, task_id: str, captcha_id: str, slot: int, target_url: Optional[str]) -> None:
        task = TaskData(
            task_id=task_id,
            status="pending",
            created_at=time.time(),
            captcha_id=captcha_id,
            slot=slot,
            target_url=target_url
        )

        async with self.lock:
            self.tasks[task_id] = task

    async def get_task(self, task_id: str) -> Optional[TaskData]:
        async with self.lock:
            return self.tasks.get(task_id)

    async def update_task(self, task_id: str, status: str, error: Optional[str] = None, 
                          result: Optional[Dict[str, Any]] = None) -> None:
        async with self.lock:
            if task_id not in self.tasks:
                return

            task = self.tasks[task_id]
            task.status = status

            if error is not None:
                task.error = error

            if result is not None:
                task.result = result

            if status in ("success", "error"):
                task.completed_at = time.time()

    async def count_tasks_by_status(self) -> Dict[str, int]:
        counts = {"pending": 0, "success": 0, "error": 0}

        async with self.lock:
            for task in self.tasks.values():
                if task.status in counts:
                    counts[task.status] += 1

        return counts

    async def cleanup_old_tasks(self) -> int:
        """Remove expired tasks"""
        now = time.time()
        tasks_to_remove = []
        expired_pending = []

        async with self.lock:
            # Find tasks to clean up
            for task_id, task in self.tasks.items():
                # Remove completed tasks after expiration
                if task.status in ("success", "error") and now - task.created_at > TASK_EXPIRE_SECONDS:
                    tasks_to_remove.append(task_id)

                # Update stale pending tasks
                elif task.status == "pending" and now - task.created_at > SOLVE_TIMEOUT * 2:
                    expired_pending.append(task_id)

            # Remove expired completed tasks
            for tid in tasks_to_remove:
                del self.tasks[tid]

            # Update stale pending tasks
            for tid in expired_pending:
                self.tasks[tid].status = "error"
                self.tasks[tid].error = "Task timed out"
                self.tasks[tid].completed_at = now

        return len(tasks_to_remove) + len(expired_pending)

    async def shutdown(self) -> None:
        logger.info("Shutting down in-memory task storage")
        # Nothing to do for in-memory storage

# =======================
#   REDIS STORAGE
# =======================
class RedisTaskStorage(TaskStorage):
    """Redis-based implementation of task storage"""

    def __init__(self, redis_url: str):
        if not REDIS_AVAILABLE:
            raise ImportError("Redis storage selected but aioredis package is not installed")

        self.redis_url = redis_url
        self.redis = None

    async def initialize(self):
        logger.info(f"Initializing Redis task storage with {self.redis_url}")
        self.redis = aioredis.from_url(
            self.redis_url,
            encoding="utf-8",
            decode_responses=True
        )
        # Test connection
        await self.redis.ping()

    async def create_task(self, task_id: str, captcha_id: str, slot: int, target_url: Optional[str]) -> None:
        task_key = f"task:{task_id}"

        task_data = {
            "status": "pending",
            "created_at": str(time.time()),
            "captcha_id": captcha_id,
            "slot": str(slot),
        }

        if target_url:
            task_data["target_url"] = target_url

        # Store task data
        await self.redis.hset(task_key, mapping=task_data)

        # Add to pending set
        await self.redis.sadd("tasks:pending", task_id)

        # Set expiration
        await self.redis.expire(task_key, TASK_EXPIRE_SECONDS)

    async def get_task(self, task_id: str) -> Optional[TaskData]:
        task_key = f"task:{task_id}"
        data = await self.redis.hgetall(task_key)

        if not data:
            return None

        created_at = float(data.get("created_at", 0))
        completed_at = float(data.get("completed_at", 0)) if "completed_at" in data else None
        slot = int(data.get("slot", 0))

        result = None
        if "result" in data:
            try:
                result = json.loads(data["result"])
            except json.JSONDecodeError:
                result = None

        return TaskData(
            task_id=task_id,
            status=data.get("status", "pending"),
            error=data.get("error"),
            result=result,
            created_at=created_at,
            completed_at=completed_at,
            captcha_id=data.get("captcha_id", ""),
            slot=slot,
            target_url=data.get("target_url")
        )

    async def update_task(self, task_id: str, status: str, error: Optional[str] = None,
                          result: Optional[Dict[str, Any]] = None) -> None:
        task_key = f"task:{task_id}"

        # Check if task exists
        exists = await self.redis.exists(task_key)
        if not exists:
            return

        # Prepare update data
        update_data = {"status": status}

        if error is not None:
            update_data["error"] = error

        if result is not None:
            update_data["result"] = json.dumps(result)

        if status in ("success", "error"):
            update_data["completed_at"] = str(time.time())

            # Move task between sets
            await self.redis.srem("tasks:pending", task_id)
            await self.redis.sadd(f"tasks:{status}", task_id)

        # Update the task
        await self.redis.hset(task_key, mapping=update_data)

    async def count_tasks_by_status(self) -> Dict[str, int]:
        counts = {}
        for status in ["pending", "success", "error"]:
            counts[status] = await self.redis.scard(f"tasks:{status}")
        return counts

    async def cleanup_old_tasks(self) -> int:
        """Clean up expired tasks in Redis"""
        cleaned_count = 0
        now = time.time()

        # Get all task keys
        keys = await self.redis.keys("task:*")

        for key in keys:
            task_id = key.split(":")[1]
            task_data = await self.redis.hgetall(key)

            if not task_data:
                continue

            created_at = float(task_data.get("created_at", 0))
            status = task_data.get("status", "")

            # If task is completed and older than expiration
            if status in ("success", "error") and now - created_at > TASK_EXPIRE_SECONDS:
                await self.redis.delete(key)
                await self.redis.srem(f"tasks:{status}", task_id)
                cleaned_count += 1

            # If task is pending but too old
            elif status == "pending" and now - created_at > SOLVE_TIMEOUT * 2:
                await self.redis.hset(key, "status", "error")
                await self.redis.hset(key, "error", "Task timed out")
                await self.redis.hset(key, "completed_at", str(now))

                await self.redis.srem("tasks:pending", task_id)
                await self.redis.sadd("tasks:error", task_id)
                cleaned_count += 1

        return cleaned_count

    async def shutdown(self) -> None:
        logger.info("Shutting down Redis task storage")
        if self.redis:
            await self.redis.close()

# =======================
#   BROWSER POOL
# =======================
class BrowserPool:
    def __init__(self, max_browsers: int = 3):
        self.max_browsers = max_browsers
        self.browsers = []
        self.available = asyncio.Queue()
        self.lock = asyncio.Lock()
        self.initialized = False
        
        # Keep a reference to the main playwright instance
        self.playwright = None

    async def initialize(self):
        if self.initialized:
            return

        # Start playwright WITHOUT using 'async with'
        self.playwright = await async_playwright().start()
        
        logger.info(f"Initializing browser pool with {self.max_browsers} browsers")
        for _ in range(self.max_browsers):
            browser = await self.playwright.chromium.launch(
                headless=False,
                args=["--disable-dev-shm-usage", "--no-sandbox"]
            )
            self.browsers.append(browser)
            await self.available.put(browser)

        self.initialized = True
        logger.info("Browser pool initialization complete")

    async def get_browser(self) -> Browser:
        # Ensure we've initialized
        if not self.initialized:
            await self.initialize()
        # Return the first available browser from the queue
        return await self.available.get()

    async def release_browser(self, browser: Browser):
        # Put the browser back
        await self.available.put(browser)

    async def shutdown(self):
        logger.info("Shutting down browser pool")

        # Close each Browser
        async with self.lock:
            while not self.available.empty():
                browser = await self.available.get()
                await browser.close()
            self.browsers.clear()
        
        # Stop the playwright instance if it's there
        if self.playwright:
            await self.playwright.stop()
            self.playwright = None

        self.initialized = False
        logger.info("Browser pool shutdown complete")

 # =======================
#   STORAGE FACTORY
# =======================
def create_task_storage() -> TaskStorage:
    """Create the appropriate storage implementation based on config"""
    if STORAGE_TYPE == "redis":
        if not REDIS_AVAILABLE:
            logger.warning("Redis storage selected but aioredis not installed. Falling back to memory storage.")
            return MemoryTaskStorage()
        return RedisTaskStorage(REDIS_URL)
    else:
        return MemoryTaskStorage()

# Global storage and browser pool instances
task_storage = create_task_storage()
browser_pool = BrowserPool(BROWSER_POOL_SIZE)

# Task semaphore to limit concurrent tasks
task_semaphore = asyncio.Semaphore(MAX_CONCURRENT_TASKS)

# =======================
#   FASTAPI APP
# =======================
app = FastAPI(
    title="Recaptcha Browser + Solver",
    description="Solves Recaptchas by spawning a headless browser + user simulation.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =======================
#  ERROR HANDLER
# =======================
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled Exception in request")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal Server Error", "error": str(exc)},
    )

# =======================
#   STARTUP/SHUTDOWN
# =======================
@app.on_event("startup")
async def startup_event():
    # Initialize storage
    await task_storage.initialize()

    # Initialize browser pool
    await browser_pool.initialize()

    # Start cleanup task
    asyncio.create_task(cleanup_old_tasks())

    logger.info(f"Server started with {STORAGE_TYPE} storage")

@app.on_event("shutdown")
async def shutdown_event():
    # Shutdown storage
    await task_storage.shutdown()

    # Shutdown browser pool
    await browser_pool.shutdown()

async def cleanup_old_tasks():
    """Background task to clean up expired tasks"""
    while True:
        try:
            count = await task_storage.cleanup_old_tasks()
            if count > 0:
                logger.info(f"Cleaned up {count} expired tasks")
        except Exception as e:
            logger.exception("Error in cleanup task")

        # Sleep before next cleanup
        await asyncio.sleep(CLEANUP_INTERVAL)

# =======================
#   ENDPOINTS
# =======================
@app.get("/health")
async def health():
    """Basic health check"""
    # Count tasks by status
    counts = await task_storage.count_tasks_by_status()

    # Check browser pool
    browser_count = len(browser_pool.browsers)
    available_browsers = browser_pool.available.qsize()

    return {
        "status": "ok",
        "storage_type": STORAGE_TYPE,
        "tasks": counts,
        "browsers": {
            "total": browser_count,
            "available": available_browsers,
        }
    }

@app.post("/submit_task")
async def submit_task(payload: SubmitTaskPayload = Body(...)):
    """Start a background job to solve a Recaptcha captcha"""
    # Check if we have capacity
    if task_semaphore.locked() and task_semaphore._value == 0:
        return JSONResponse(
            status_code=429,
            content={
                "detail": "Too many concurrent tasks. Please try again later.",
                "retry_after": 5
            }
        )

    task_id = str(uuid.uuid4())

    # Create the task
    await task_storage.create_task(
        task_id=task_id,
        captcha_id=payload.captcha_id,
        slot=payload.slot,
        target_url=payload.target_url
    )

    # Spawn the background task
    asyncio.create_task(_solve_task(task_id, payload.captcha_id, payload.slot, payload.target_url))

    return {"task_id": task_id, "status": "pending"}

@app.post("/get_result")
async def get_result(payload: GetResultPayload = Body(...)):
    """Poll for results of the puzzle solve"""
    task_id = payload.task_id

    # Get task data
    task = await task_storage.get_task(task_id)

    if not task:
        raise HTTPException(status_code=404, detail="No such task_id.")

    if task.status == "pending":
        return {
            "task_id": task_id,
            "status": "pending",
            "message": "Still solving...",
            "created_at": task.created_at
        }
    elif task.status == "success":
        return {
            "task_id": task_id,
            "status": "success",
            "result": task.result,
            "created_at": task.created_at,
            "completed_at": task.completed_at
        }
    else:  # status == "error"
        return {
            "task_id": task_id,
            "status": "error",
            "error": task.error,
            "created_at": task.created_at,
            "completed_at": task.completed_at
        }

# ========================
#  BACKGROUND TASK
# ========================
async def _solve_task(task_id: str, captcha_id: str, slot: int, target_url: Optional[str]):
    """Runs in the background to solve the captcha"""
    start_time = time.time()

    # Acquire semaphore to limit concurrent tasks
    async with task_semaphore:
        try:
            # Determine the page URL
            if target_url:
                page_url = target_url
            else:
                page_url = f"http://localhost:8077/{slot}"

            # Actually run the logic
            result_data = await solve_recaptcha_in_browser(page_url, captcha_id)

            # Update task status to success
            await task_storage.update_task(
                task_id=task_id,
                status="success",
                result=result_data
            )

        except Exception as e:
            logger.exception(f"Exception in solve task {task_id}")

            # Update task status to error
            await task_storage.update_task(
                task_id=task_id,
                status="error",
                error=str(e)
            )

        finally:
            elapsed = time.time() - start_time
            logger.info(f"[Task {task_id}] Done in {elapsed:.2f}s")

# ========================
#  CORE BROWSER LOGIC
# ========================
def format_result(success: bool, data: Any, start_time: float) -> Dict[str, Any]:
    """
    Helper that standardizes the solver's return dictionary and logs the outcome.
    """
    elapsed = time.time() - start_time
    result = {
        "success": success,
        "data": data,
        "time": round(elapsed, 3),
    }

    if success:
        logger.info(f"[success - Elapsed: {elapsed:.3f}s")
    else:
        logger.warning(f"[failure - Elapsed: {elapsed:.3f}s")

    return result

async def solve_recaptcha_in_browser(page_url: str, captcha_id: str) -> Dict[str, Any]:
    """
    Solve Recaptchas by automating a browser
    """
    import json
    import random

    result={}
    input_data = {'type': '', 'sitekey': ''}

    # Get a browser from the pool
    browser = await browser_pool.get_browser()

    try:
        # Create new context for isolation
        context = await browser.new_context(
            viewport={"width": 1280, "height": 800},
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/91.0.4472.124 Safari/537.36"
            )
        )

        # Set up request interception
        def solve(**kwargs) -> Dict[str, Any]:
            start_time = time.time()
            bg_url: str = kwargs.get("bg_url", "")
            site_key: str = kwargs.get("site_key", "")
            type: str = kwargs.get("type", "")
            api_key: str = kwargs.get("api_key", "")
            info = {"page url": bg_url, "site key": site_key, "api key": api_key}
            for key in info.keys():
                if not info[key]:
                    return format_result(False, f"Missing{key}.", start_time)
            try:
                # instantiate a captchaai solver
                solver = CaptchaAI(api_key)
                if type=='v3':
                    result = solver.recaptcha(
                        sitekey=site_key,
                        url=bg_url,
                        version='v3',
                    )

                elif type=='invisible':
                    result = solver.recaptcha(
                        sitekey=site_key,
                        url=bg_url,
                        invisible=1)
                else:
                    result = solver.recaptcha(
                        sitekey=site_key,
                        url=bg_url)
                logger.info('solution received: '+str(result))
                return format_result(True, result, start_time)

            except Exception as ex:
                print('solve error',ex)
                raise Exception(f"failed to get the solution: {str(e)}")

        async def handle_request(request):
            try:
                if "google.com/recaptcha" in request.url and 'k=' in request.url:
                    if not input_data['sitekey']:
                        site_key = request.url.split('&k=')[1].split('&')[0]
                        input_data['sitekey']=site_key
                        print('handle request got site key',site_key)
                        if 'size=invisible' not in request.url:
                            input_data['type'] = 'v2'
                        else:
                            input_data['type'] = 'invisible'
            except Exception as e:
                pass
        # Set up event handler
        context.on("request", handle_request)
        logger.info('INPUT DATA: ', input_data)
        # Create new page
        page = await context.new_page()

        # Add better error logging
        page.on("console", lambda msg: logger.info(f"BROWSER CONSOLE: {msg.text}"))
        page.on("pageerror", lambda err: logger.error(f"BROWSER PAGE ERROR: {err}"))

        # Navigate with retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                logger.info(f"Navigating to {page_url} (attempt {attempt+1})")
                await page.goto(page_url, timeout=30000, wait_until="networkidle")
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                logger.warning(f"Navigation failed, retrying: {str(e)}")
                await asyncio.sleep(2)




        if input_data['type']=='invisible' or not input_data['type']:
            try:
                div = page.locator('//div[@class="g-recaptcha"]')
            except Exception as e:
                logger.info("no g-recaptcha div so this recaptcha type will be considered v3"+str(e))
                input_data['type']='v3'
        # Wait for recaptcha to load
        print(input_data)
        if not input_data['sitekey']:
            try:
                logger.info("getting site key from site")
                try:
                    input_data['sitekey'] = await div.get_attribute('data-sitekey')
                    logger.info(f"site key is: {input_data['sitekey']}")
                except Exception as e:
                    screenshot_path = f"error-sitekey-{int(time.time())}.png"
                    logger.error('failed to get site key',e)
                    await page.screenshot(path=screenshot_path)
                    raise Exception("Did not capture the site key")

            except Exception as e:
                logger.error(f"recaptcha div not there: {str(e)}")
                screenshot_path = f"error-button-{int(time.time())}.png"
                await page.screenshot(path=screenshot_path)
                raise Exception(f"failed to get the site key: {str(e)}")

        result = solve(bg_url=page_url, site_key=input_data['sitekey'],type=input_data['type'], api_key=API_KEY)
        print('result',result)
        # c = requests.post("https://www.google.com/recaptcha/api/siteverify",
        #                   data={'secret': site_key, "response": result['code']})
        success='fail'
        if result["success"]:

            data = result['data']
            if input_data['type'] == 'v3':
                try:
                    js = f"window.verifyRecaptcha('" + data['code'] + "');"
                    await page.evaluate(js)
                    success='success'
                except Exception as e:
                    raise Exception(f"Failed to verify on site: {str(e)}")
            else:
                try:
                    xpath = '//textarea[contains(@id,"g-recaptcha-response")]'
                    WE = await page.query_selector(xpath)
                    js = f"WE => WE.setAttribute('style','')"
                    await page.evaluate(js, WE)
                    if input_data['type'] == 'invisible':
                        try:
                            xpath = '//div[@class="grecaptcha-badge"]'
                            WE1 = await page.query_selector(xpath)
                            await page.evaluate(js, WE1)
                        except Exception as e:
                            print(e)

                    try:
                        await WE.fill(data['code'])
                        success='success'
                    except Exception as e:
                        raise Exception(f"Failed to fill the recaptcha textarea with the result code: {str(e)}")
                    await asyncio.sleep(120)
                except Exception as e:
                    raise Exception(f"Failed solve on site: {str(e)}")
            return {'result': success}
            # try:
            #     c = requests.post(
            #     "https://www.google.com/recaptcha/api2/userverify?k=" + site_key)
            #     print(1,c.text)
            # except Exception as e:
            #     print(e)
        else:
            return {'result': 'fail','error':'No result data'}

    finally:
        # Always clean up and return browser to pool
        try:
            if 'context' in locals():
                await context.close()
        except Exception as e:
            logger.error(f"Error closing browser context: {str(e)}")

        await browser_pool.release_browser(browser)

# ========================
#  MAIN ENTRYPOINT
# ========================
if __name__ == "__main__":
    # import importlib.metadata
    # # Log environment info
    # logger.info(f"Python version: {sys.version}")
    # logger.info(f"FastAPI version: {importlib.metadata.version('fastapi')}")
    # logger.info(f"Uvicorn version: {importlib.metadata.version('uvicorn')}")

    # For local debugging:
    # Single process with auto-reload for development
    uvicorn.run(
        "main:app", 
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "9080")),
        reload=os.getenv("RELOAD", "false").lower() == "true"
    )
    
    # For production:
    # Use Gunicorn with Uvicorn workers:
    # gunicorn -w 4 -k uvicorn.workers.UvicornWorker main:app --bind 0.0.0.0:9080