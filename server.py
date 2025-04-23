# server.py
import os
import json
import hmac
import hashlib
import tornado.web
import tornado.ioloop
import tornado.httpserver
import tornado.process
import logging

logger = logging.getLogger(__name__)
CAPTCHA_ID = os.getenv("CAPTCHA_ID", "e392e1d7fd421dc63325744d5a2b9c73")
CAPTCHA_KEY = os.getenv("CAPTCHA_KEY", "59bbf667d8c11320efbea3cc538ca3c4")
API_SERVER = os.getenv("API_SERVER", "http://gcaptcha4.geetest.com")

class MainHandler(tornado.web.RequestHandler):
    def get(self):
        try:
            # Renders a static HTML file
            self.render("index.html")
        except ValueError:
            self.send_error(400, message="Invalid number in URL")
        except Exception as e:
            logger.exception("Error in MainHandler")
            self.send_error(500, message=str(e))
            
class LoginHandler(tornado.web.RequestHandler):
    async def post(self):
        try:
            # Try to decode JSON body; fallback to form arguments if needed
            try:
                data = json.loads(self.request.body.decode("utf-8"))
            except Exception:
                data = {}
            lot_number = data.get('lot_number', self.get_argument('lot_number', ''))
            captcha_output = data.get('captcha_output', self.get_argument('captcha_output', ''))
            pass_token = data.get('pass_token', self.get_argument('pass_token', ''))
            gen_time = data.get('gen_time', self.get_argument('gen_time', ''))

            if not lot_number:
                self.set_status(400)
                self.write({"login": "fail", "reason": "Missing lot_number"})
                return

            # Generate HMAC-SHA256 signature using lot_number and secret key
            sign_token = hmac.new(
                CAPTCHA_KEY.encode(),
                lot_number.encode(),
                digestmod=hashlib.sha256  # Use the actual algorithm object
            ).hexdigest()

            query = {
                "lot_number": lot_number,
                "captcha_output": captcha_output,
                "pass_token": pass_token,
                "gen_time": gen_time,
                "sign_token": sign_token,
            }

            # Here you would normally send an asynchronous HTTP request to the GeeTest API.
            # For demonstration, we simulate a successful verification.
            # TODO: Implement actual verification with GeeTest API
            self.write({"login": "success", "reason": "Verified"})
        except Exception as e:
            logger.exception("Error during captcha login verification")
            self.set_status(500)
            self.write({"login": "fail", "reason": str(e)})

def make_app():
    return tornado.web.Application([
        (r'/', MainHandler),  # Handler for root path
        (r'/[0-9]+', MainHandler),
        (r'/login', LoginHandler),
    ],
    debug=False,
    template_path=os.path.join(os.path.dirname(__file__), "templates"),
    static_path=os.path.join(os.path.dirname(__file__), "static"),
    static_url_prefix="/static/")

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    app = make_app()
    server = tornado.httpserver.HTTPServer(app)
    port = int(os.getenv("PORT", "8077"))
    server.bind(port)
    
    # For production: Uncomment this to enable multi-processing
    # num_processes = int(os.getenv("NUM_PROCESSES", "0"))  # 0 means auto-detect CPU count
    # tornado.process.fork_processes(num_processes)
    
    logger.info(f"Server starting on port {port}.")
    server.start()  # Use start() instead of single-process IOLoop
    tornado.ioloop.IOLoop.current().start()