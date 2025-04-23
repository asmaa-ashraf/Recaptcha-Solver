# Recaptcha-Solver
This project uses the services of Capatchaai library to solve recaptchas on site 
An API is created to receive requests containing the website that contains the recaptcha to be solved.
then the program instantiates a playwright browser, gets the page, gets the required information from the page
and sends them to captchaai then waits for the result then it does what is required to send the solution of the recaptcha to the page
