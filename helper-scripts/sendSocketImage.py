import cv2
import struct
import socket
import numpy as np

vc = cv2.VideoCapture(0)

ret, frame = vc.read()

def sendImageThroughSocket(socket,image):
    img_str = cv2.imencode('.jpg', image)[1].tostring()
    imageSize = len(img_str)
    val = struct.pack('!i', imageSize)
    socket.send(val)
    socket.send(img_str)
    print imageSize

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#s.connect(('10.12.5.208', 5555))
s.connect(('localhost', 8000))

sendImageThroughSocket(s,frame)
cv2.waitKey(1000)