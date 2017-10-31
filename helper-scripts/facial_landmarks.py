# import the necessary packages
import sys
import ctypes
import os
# lib = ctypes.CDLL(os.path.join('/home/alex/anaconda2/lib/', 'libboost_python.so.1.61.0'))
#
# lib2 = ctypes.CDLL(os.path.join('/home/alex/anaconda2/lib/', 'libpng16.so.16'))
# lib3 = ctypes.CDLL(os.path.join('/home/alex/anaconda2/lib/', 'libz.so.1'))
sys.path.append('/home/alex/anaconda2/lib/python2.7/site-packages/dlib-19.6.0-py2.7-linux-x86_64.egg')
sys.path.append('/home/alex/anaconda2/lib/python2.7/site-packages/')
sys.path.append('/home/alex/anaconda2/lib/python2.7/site-packages/')
sys.path.append('/home/alex/anaconda2/lib/')
sys.path.append('/home/alex/anaconda2/lib/python2.7')
from imutils.video import VideoStream
from imutils import face_utils
import datetime
import argparse
import imutils
import time
import dlib
import cv2
################################### FUCKING ADD the environment variable LD_LIBRARY_PATH for the path when it does not find shared libraries

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
                help="path to facial landmark predictor")
ap.add_argument("-r", "--picamera", type=int, default=-1,
                help="whether or not the Raspberry Pi camera should be used")
args = vars(ap.parse_args())

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
print args["shape_predictor"]
predictor = dlib.shape_predictor(args["shape_predictor"])

# initialize the video stream and allow the cammera sensor to warmup
print("[INFO] camera sensor warming up...")
time.sleep(2.0)


# import the necessary packages
from imutils import face_utils
import datetime
import argparse
import imutils
import time
import dlib
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
                help="path to facial landmark predictor")
ap.add_argument("-r", "--picamera", type=int, default=-1,
                help="whether or not the Raspberry Pi camera should be used")
args = vars(ap.parse_args())

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# initialize the video stream and allow the cammera sensor to warmup
print("[INFO] camera sensor warming up...")
#vs = VideoStream(usePiCamera=args["picamera"] > 0).start()
vc = cv2.VideoCapture(0)
time.sleep(2.0)
faceCascade = cv2.CascadeClassifier('/home/alex/Downloads/opencv-3.1.0/data/haarcascades/haarcascade_frontalface_alt.xml')
# loop over the frames from the video stream
while True:
    # grab the frame from the threaded video stream, resize it to
    # have a maximum width of 400 pixels, and convert it to
    # grayscale
    #frame = vs.read()
    stat, frame = vc.read()
    #frame = cv2.resize(frame,(640,480), interpolation = cv2.INTER_CUBIC)
    # frame = imutils.resize(frame, width=400)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # faces = faceCascade.detectMultiScale(
    #     gray,
    #     scaleFactor=1.1,
    #     minNeighbors=5,
    #     minSize=(30, 30),
    #     flags =cv2.CASCADE_SCALE_IMAGE
    # )
    # detect faces in the grayscale frame
    rects = detector(gray, 0)
    for rect in rects:
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # loop over the (x, y)-coordinates for the facial landmarks
        # and draw them on the image
        for (x, y) in shape:
            cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

        # show the frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break
cv2.destroyAllWindows()
vs.stop()