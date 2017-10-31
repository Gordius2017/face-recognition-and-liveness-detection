import socket
import struct
import numpy as np
import cv2
import requests

test_url = 'http://10.5.5.39:8080/api/base64/image'
content_type = 'image/jpeg'
headers = {'content-type': content_type}



def getImageFromSocket(s,colorCode = cv2.IMREAD_GRAYSCALE,crop = True): #or cv2.IMREAD_COLOR


    print "received connection"

    imageSize = s.recv(4)
    imageSize = struct.unpack('i', imageSize)[0]
    imageBytes = b''
    while imageSize > 0:
        chunk = s.recv(imageSize)
        imageBytes += chunk
        imageSize -= len(chunk)

    data = np.fromstring(imageBytes, dtype='uint8')

    response = requests.post(test_url, data=data.tostring().encode("base64"), headers=headers)
    print response.text

    image = cv2.imdecode(data, colorCode)
    if crop:
        image = image[230:480, 0:640]
    return image

serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
serversocket.bind(('localhost', 8001))
serversocket.listen(5)
(clientsocket, address) = serversocket.accept()
image = getImageFromSocket(clientsocket)