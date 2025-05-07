import json

from captchaai import CaptchaAI
from playwright.async_api import async_playwright, Page, Browser, BrowserContext
from playwright.async_api import TimeoutError as PlaywrightTimeout, expect
from typing import Dict, Any, Optional, List, Union
import logging
import asyncio
import time
import os
import random
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


API_KEY = os.getenv("API_KEY", 'b755df7ff3dea3991eaf74c10882cdfc')
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

def data_generator():
    name = random.choice(['alan','carry','gumball','anaiis','richard','watterson'])+ str(random.randint(111,999))
    email = name + str(random.randint(111,999))+'.'+ str(random.randint(111,999))+'@yahoo.com'
    return{'name':name, 'email':email, 'password':'newestPassword123#'}

async def solve_recaptcha_in_browser(page_url: str, captcha_type: str, site_data=[]):
    """
    Solve Recaptchas by automating a browser
    """
    import json
    import random
    result = {}
    input_data = {'type': captcha_type, 'sitekey': ''}
    success,time_taken,err='',0,''
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
            try:
                # instantiate a captchaai solver
                solver = CaptchaAI(api_key)
                if type == 'v3':
                    result = solver.recaptcha(
                        sitekey=site_key,
                        url=bg_url,
                        version='v3',
                    )
                elif type == 'invisible':
                    result = solver.recaptcha(
                        sitekey=site_key,
                        url=bg_url,
                        invisible=1)
                else:
                    result = solver.recaptcha(
                        sitekey=site_key,
                        url=bg_url)
                print('solution',result)
                end_time=time.time()
                time_taken = end_time - start_time
                print('time taken',time_taken)
                logger.info('solution received: ' + str(result))
                return result, time_taken

            except Exception as ex:
                print('solve error', ex)
                err = str(ex)
                raise Exception(f"failed to get the solution: {str(ex)}")

        async def handle_request(request):
            print(request.url)
            try:
                if request.url=="https://www.google.com/recaptcha/api.js" or "recaptcha/verify" in request.url:

                    try:
                        print(request.response)
                        print(request.text)
                        print(request.content)
                        print(request.json())
                    except:
                        pass

                if "google.com/recaptcha" in request.url and 'k=' in request.url:
                    if not input_data['sitekey']:
                        site_key = request.url.split('&k=')[1].split('&')[0]
                        input_data['sitekey'] = site_key
                        print('handle request got site key', site_key)
                        # if 'size=invisible' not in request.url:
                        #     input_data['type'] = 'v2'
                        # else:
                        #     input_data['type'] = 'invisible'
            except:
                pass

        # Set up event handler
        context.on("request", handle_request)
        # Create new page
        page = await context.new_page()

        # Add better error logging
        page.on("console", lambda msg: logger.info(f"BROWSER CONSOLE: {msg.text}"))
        page.on("pageerror", lambda err: logger.error(f"BROWSER PAGE ERROR: {err}"))

        # Navigate with retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                logger.info(f"Navigating to {page_url} (attempt {attempt + 1})")
                await page.goto(page_url, timeout=30000, wait_until="networkidle")
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                logger.warning(f"Navigation failed, retrying: {str(e)}")
                await asyncio.sleep(2)

        if input_data['type'] == 'invisible' :
            try:
                div = page.locator('//div[@class="g-recaptcha"]')
            except Exception as e:
                logger.info("no g-recaptcha div so this recaptcha type will be considered v3" + str(e))
                # input_data['type'] = 'v3'
        # Wait for recaptcha to load
        print(input_data)
        x=0
        while not input_data['sitekey'] and x<20:
            print('waiting for site key from api requests')
            x+=1
            await asyncio.sleep(2)
        if not input_data['sitekey']:
            raise Exception(f"USER_ERROR: failed to get site key")
        result,time_taken = solve(bg_url=page_url, site_key=input_data['sitekey'], type=input_data['type'], api_key=API_KEY)
        print('result', result)
        # c = requests.post("https://www.google.com/recaptcha/api/siteverify",
        #                   data={'secret': site_key, "response": result['code']})
        success = 0
        if result:

            if input_data['type'] == 'v3':
                try:
                    js = f"window.verifyRecaptcha('" + result['code'] + "');"
                    await page.evaluate(js)
                    print('code sent')
                    await asyncio.sleep(20)
                    success = 1
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
                            print(264,e)

                    try:
                        await WE.fill(result['code'])
                        success = 1
                        print('code sent')
                    except Exception as e:
                        raise Exception(f"Failed to fill the recaptcha textarea with the result code: {str(e)}")

                except Exception as e:
                    raise Exception(f"Failed solve on site: {str(e)}")
                await asyncio.sleep(10)
        else:
            success= 0
        if site_data and success:
            test_WE = ''
            for ele in site_data:
                if ele=='inputs':
                    inputs = site_data['inputs']
                    new_data = data_generator()
                    for input in inputs:
                        print(input)
                        if 'email' in input:
                            d = new_data['email']
                        elif 'password' in input or 'pwd' in input:
                            d = new_data['password']
                        else:
                            d = new_data['name']
                        WE = await page.query_selector(input)
                        test_WE = WE
                        await WE.fill(d)
                        print(d,'sent')
                        await asyncio.sleep(5)

                if 'clicks' in ele:
                    clicks = site_data[ele]
                    for click in clicks:
                        print(click)
                        WE = await page.query_selector(click)
                        await WE.click()
                        await asyncio.sleep(25)
                        if not test_WE:
                            test_WE = WE
                        print('clicked')
            try:
                await test_WE.click()
                print( 'test_we still there indicating failure')
                success=0
            except Exception as e:
                print('testing success of code sending',e)
                success=1

    except Exception as e:
        err=str(e)
        print(325,e)
    finally:
        # Always clean up and return browser to pool
        try:
            if 'context' in locals():
                await context.close()
        except Exception as e:
            logger.error(f"Error closing browser context: {str(e)}")

        await browser_pool.release_browser(browser)
        return success,time_taken,err

async def rater(url,site_data):
    try:
        err =0
        s=0
        captcha_type=site_data['type']
        print(captcha_type)
        try:
            s,time_taken,err =await solve_recaptcha_in_browser(url,captcha_type,site_data)
        except Exception as e:
            print(319,e)
        try:
            with open(captcha_type + '_rates.txt', 'r') as f:
                text = f.read()
                rates = json.loads(text)
        except:
            rates={'success':0,'fail':0,'success_rate':0,'time_medium':0.0,'urls':[url]}
        if url not in rates['urls']:
            rates['urls'].append(url)
        if err:
            if err in rates:
                rates[err] += 1
            else:
                rates[err] = 1
        if s:
            rates['success'] +=1
            print(rates['success'],'success rate')
        elif 'USER_ERROR' not in err: #if user error then it shouldn't be added to the rating but the error will be recorded:
            rates['fail'] += 1
            print(rates['fail'], 'fail rate')
        times_tested = rates['fail']+rates['success']
        if times_tested:
            rates['success_rate'] = 100 * rates['success']/times_tested
        if rates['time_medium']:
            rates['time_medium'] = (rates['time_medium']+ time_taken) / 2
        else:
                rates['time_medium'] = time_taken
        with open(captcha_type + '_rates.txt','w')as f:
            f.write(json.dumps(rates))
    except Exception as e:
        print(e)
v3_data={'clicks':'xpath=//button[text()="check"]','type':'v3'}
v2_data = {'inputs': ['//input[@name="email"]', '//input[@name="alias"]', '//input[@type="password"]'],
             'clicks': ['//button[@id="signup"]'],'type':'v2',"https://squareblogs.net/signup":'v2_data'}
github_v3={'type':'v3'}
invsbl_data = {'clicks':['xpath=//button[text()="check"]'],'type':'invisible'}
urls = { "https://testrecaptcha.github.io/":github_v3,"https://2captcha.com/demo/recaptcha-v3":v3_data,
        "https://2captcha.com/demo/recaptcha-v2-invisible":invsbl_data}
cycles=10
for cycle in range(cycles):
    for url in urls.keys():
        print(url)
        browser_pool = BrowserPool(1)
        asyncio.run(rater(url,urls[url]))