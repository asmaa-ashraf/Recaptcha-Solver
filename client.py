import requests
from time import sleep
import json
URL="http://127.0.0.1:9080"
target_url='https://squareblogs.net/signup'
target_url="https://blogfreely.net/"
invisiblecaptchaurl='https://en.over-blog.com/'
v3url = 'https://2captcha.com/demo/recaptcha-v3'
urls={'invisible':invisiblecaptchaurl, 'v2':target_url, 'v3': v3url}

#Testing api endpoints with different types of recaptcha

#/health endpoint when there are no tasks
request=requests.get(URL+"/health")
print('request get /health before sending any task:')
print(request.text)

#sending /submit_task request for three sites with different types of recaptcha
task_ids={}
tasks_ids=["5baa21ed-b894-4a2d-9908-98c045fd32f0","4d7682fe-8161-4465-a241-f025f1020bba","c0e93305-cb2f-4181-ae4c-d2bc537c86ca"]
for type in urls.keys():
    url=urls[type]
    print(f"sending /submit_task request for target url: {urls[type]} that has a recaptcha of type {type}")
    request= requests.post(URL+"/submit_task",json={'captcha_id':'', 'slot':0, 'target_url':url})
    print(request.text)
    j = json.loads(request.text)
    task_ids[url] = j['task_id']
    break
#/health endpoint after sending tasks
request = requests.get(URL+"/health")
print('request get /health after sending tasks:')
print(request.text)
sleep(30)
for url in task_ids.keys():
    break
    while True:
        task_id=task_ids[url]
        print(f'posting /get_result request for site {url}')
        request = requests.post(URL+"/get_result", json={'task_id': task_id})
        print(request.text)
        j = json.loads(request.text)
        status= j['status']
        if status != 'pending':
            break
        sleep(5)