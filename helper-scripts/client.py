import cv2
import struct
import socket
import numpy as np

def getImageFromSocket(s):
    imageSize = s.recv(4)
    imageSize = imageSize[::-1]
    imageSize = struct.unpack('i', imageSize)[0]
    imageBytes = b''
    while imageSize > 0:
        chunk = s.recv(imageSize)
        imageBytes += chunk
        imageSize -= len(chunk)

    data = np.fromstring(imageBytes, dtype='uint8')
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return image

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#s.connect(('10.12.5.208', 5555))
s.connect(('localhost', 5555))
#86.120.51.250
while True:
    image = getImageFromSocket(s)
    cv2.imshow("image",image)
    cv2.waitKey(20)