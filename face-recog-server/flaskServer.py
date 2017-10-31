from flask import Flask, jsonify, request
import threading
import socket
import struct
import numpy as np
import cv2
import requests

faceCascade = cv2.CascadeClassifier('/home/alex/Downloads/opencv-3.1.0/data/haarcascades/haarcascade_frontalface_alt.xml')

global imageRPI
global face_recognizer

app = Flask(__name__)

def detect_face(img,face_cascade):
    #convert the test image to gray scale as opencv face detector expects gray images
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
    if (len(faces) == 0):
        return None, None
    (x, y, w, h) = faces[0]
    return gray[y:y+w, x:x+h], faces[0]

# def process_image(image):
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags =cv2.CASCADE_SCALE_IMAGE)
#     faces_images = []
#     for (x, y, w, h) in faces:
#         faces_images.append(image[y:y+h,x:x+w])
#         cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
#     cv2.imshow("Faces found" ,image)
#     cv2.waitKey(10)
def predict_img(test_img,face_recognizer):

    #make a copy of the image as we don't want to change original image
    if test_img is None:
        return None
    img = test_img.copy()
    #detect face from the image
    face, rect = detect_face(img,faceCascade)
    label_text = None
    #predict the image using our face recognizer
    if face is not None:
        label, conf= face_recognizer.predict(face)
    #get name of respective label returned by face recognizer
        if label is not -1 and conf < 80:
            label_text = [line.rstrip('\n') for line in open("training_data/s"+str(label)+"/name.txt", "r")]

        else:
            label_text = ["Unknown"]

    #draw a rectangle around face detected
        draw_rectangle(img, rect)
        #draw name of predicted person
        draw_text(img, label_text[0]+str(conf), rect[0], rect[1]-5)

    return img,label_text


def getImageFromSocket():

    global imageRPI
    global face_recognizer
    face_recognizer= cv2.face.LBPHFaceRecognizer_create()

    face_recognizer.read('face_recognizer.model')
    serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    serversocket.bind(('localhost', 8001))
    serversocket.listen(5)
    (clientsocket, address) = serversocket.accept()

    from requests_futures.sessions import FuturesSession
    session = FuturesSession()
    session2 = FuturesSession()
    contor = 0

    while True:
        imageSize = clientsocket.recv(4)
        imageSize = struct.unpack('i', imageSize)[0]
        imageBytes = b''
        while imageSize > 0:
            chunk = clientsocket.recv(imageSize)
            imageBytes += chunk
            imageSize -= len(chunk)

        data = np.fromstring(imageBytes, dtype='uint8')
        imageRPI = cv2.imdecode(data, cv2.IMREAD_COLOR)
        img,label_text = predict_img(imageRPI,face_recognizer)
        if img is not None :
            cv2.imshow("image",img)
            cv2.waitKey(1)
        elif imageRPI is not None:
            cv2.imshow("image",imageRPI)
            cv2.waitKey(1)

        contor = contor +1
        if contor % 5 ==0:
            imageBytes = cv2.imencode('.jpg',img,[int(cv2.IMWRITE_JPEG_QUALITY), 16])[1].tostring()
            #grequests.post('http://10.5.5.39:8080/api/capture/frame',imageBytes.encode("base64"))
            session.post('http://10.5.5.39:8080/api/capture/frame',imageBytes.encode("base64"))
            if label_text is not None:
                with app.test_request_context():
                    from flask import request
                    session.post('http://10.5.5.39:8080/notification/updatedetails', data=label_text[0])
            #future = pool.submit(requests.post, ('http://10.5.5.39:8080/api/capture/frame',imageBytes.encode("base64")))
            #response = requests.post('http://10.5.5.39:8080/api/capture/frame', data=imageBytes.encode("base64"))
            #print response.text

threading.Thread(target=getImageFromSocket).start()


from flask.ext.cors import CORS, cross_origin
app.config['SECRET_KEY'] = 'the quick brown fox jumps over the lazy   dog'
app.config['CORS_HEADERS'] = 'Content-Type'

cors = CORS(app)
import os
@app.route('/addNewUser',methods=['POST'])
@cross_origin(origin='localhost',headers=['Content- Type','Authorization'])
def addNewUser():
    global imageRPI
    imageRPI = cv2.imread("/home/alex/Desktop/1.jpg")
    data = request.json
    id = str(data['id'])
    user_name = data['name']
    try:
        os.mkdir('training_data/'+'s'+id)
        text_file = open("name.txt", "w")
        text_file.write(user_name)
        text_file.close()
        for x in range(50):
            cv2.imwrite("training_data/"+'s'+id+"/"+"img"+str(x)+".jpg",imageRPI)
            cv2.waitKey(250)
    except:
        return jsonify(resp='User id allready exist')
    return jsonify(resp='New User adding')# ls cold be any word



def prepare_training_data():
    data_folder_path = 'training_data'
    dirs = os.listdir(data_folder_path)
    face_cascade = cv2.CascadeClassifier('/home/alex/Downloads/opencv-3.1.0/data/haarcascades/haarcascade_frontalface_alt.xml')
    faces = []
    labels = []
    for dir_name in dirs:
        if not dir_name.startswith("s"):
            continue
        label = int(dir_name.replace("s", ""))
        subject_dir_path = data_folder_path + "/" + dir_name
        subject_images_names = os.listdir(subject_dir_path)

        for image_name in subject_images_names:
            if image_name.startswith("."):
                continue
            if image_name == 'name.txt':
                continue
            image_path = subject_dir_path + "/" + image_name
            print image_path
            image = cv2.imread(image_path)
            face, rect = detect_face(image,face_cascade)
            if face is not None:
                faces.append(face)
                labels.append(label)
    return faces, labels

@app.route("/train",methods=['GET'])
def train():
    print("Preparing data...")
    faces, labels = prepare_training_data()
    global face_recognizer
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.train(faces, np.array(labels))
    face_recognizer.write("face_recognizer.model")

    return "hello"

def draw_rectangle(img, rect):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

def draw_text(img, text, x, y):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)

from scipy.spatial import distance as dist
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

import dlib
from imutils import face_utils
EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 3
global COUNTER
global TOTAL
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

def blink_counter():
    global imageRPI
    global COUNTER
    global TOTAL
    imageRPI = None
    COUNTER = 0
    TOTAL = 0
    while True:

        if imageRPI is None:
            cv2.waitKey(100)
            continue
        print "processing"
        blink_image = imageRPI.copy()
        gray = cv2.cvtColor(blink_image, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 0)
        for rect in rects:
            # determine the facial landmarks for the face region, then
            # convert the facial landmark (x, y)-coordinates to a NumPy
            # array
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            # extract the left and right eye coordinates, then use the
            # coordinates to compute the eye aspect ratio for both eyes
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)

            # average the eye aspect ratio together for both eyes
            ear = (leftEAR + rightEAR) / 2.0
            # compute the convex hull for the left and right eye, then
            # visualize each of the eyes
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(blink_image, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(blink_image, [rightEyeHull], -1, (0, 255, 0), 1)
            # check to see if the eye aspect ratio is below the blink
            # threshold, and if so, increment the blink frame counter
            if ear < EYE_AR_THRESH:
                COUNTER += 1

            # otherwise, the eye aspect ratio is not below the blink
            # threshold
            else:
                # if the eyes were closed for a sufficient number of
                # then increment the total number of blinks
                if COUNTER >= EYE_AR_CONSEC_FRAMES:
                    TOTAL += 1

                # reset the eye frame counter
                COUNTER = 0
            # draw the total number of blinks on the frame along with
            # the computed eye aspect ratio for the frame

            cv2.putText(blink_image, "Blinks: {}".format(TOTAL), (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(blink_image, "EAR: {:.2f}".format(ear), (300, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow("blink image", blink_image)
        cv2.waitKey(10)


#threading.Thread(target=blink_counter).start()

import io
@app.route("/predict",methods=['POST'])
def predict():
    face_recognizer = cv2.face.createFisherFaceRecognizer()
    face_recognizer.load("face_recognizer.model")
    if request.method == 'POST':
        photo = request.files['file']
        in_memory_file = io.BytesIO()
        photo.save(in_memory_file)
        data = np.fromstring(in_memory_file.getvalue(), dtype=np.uint8)
        color_image_flag = 1
        img = cv2.imdecode(data, color_image_flag)
        face, rect = detect_face(img)
        label= face_recognizer.predict(face)
        lines = [line.rstrip('\n') for line in open("training_data/s"+str(label)+"name.txt", "r")]
        print lines[0]
        return jsonify(person_name=lines[0])
    return jsonify(person_name='Unknown')


if __name__ == '__main__':
    app.run()

