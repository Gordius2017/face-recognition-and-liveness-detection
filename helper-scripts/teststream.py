import socket
import struct
import numpy as np
import cv2


serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
serversocket.bind(('10.5.5.39', 5555))
#192.168.0.102
serversocket.listen(5)

(clientsocket, address) = serversocket.accept()
video_capture = cv2.VideoCapture(0)
def sendImageThroughSocket(socket,image):
    img_str = cv2.imencode('.jpg', image)[1].tostring()
    imageSize = len(img_str)
    val = struct.pack('!i', imageSize)
    socket.send(val)
    socket.send(img_str)
    print imageSize

while True:
    ret,image = video_capture.read()
    sendImageThroughSocket(clientsocket,image)