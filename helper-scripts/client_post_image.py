import requests
import json
import cv2

addr = 'http://5.12.39.110:8080/api/base64/image'
test_url = addr

# prepare headers for http request
content_type = 'image/jpeg'
headers = {'content-type': content_type}


import cv2
video_capture = cv2.VideoCapture(0)
while True:
    ret,img = video_capture.read()
    # encode image as jpeg
    _, img_encoded = cv2.imencode('.jpg', img)
    # send http request with image and receive response
    #response = requests.post(test_url, data=img_encoded.tostring(), headers=headers)
    response = requests.post(test_url, data=img_encoded.tostring().encode("base64"), headers=headers)

    # decode response
    print response.text

