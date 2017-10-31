import cv2

faceCascade = cv2.CascadeClassifier('/home/alex/Downloads/opencv-3.1.0/data/haarcascades/haarcascade_frontalface_alt.xml')
eyeCascade = cv2.CascadeClassifier('/home/alex/Downloads/opencv-3.1.0/data/haarcascades/haarcascade_eye.xml')
smileCascade = cv2.CascadeClassifier('/home/alex/Downloads/opencv-3.1.0/data/haarcascades/haarcascade_smile.xml')


# image = cv2.imread("crowd_2.jpg")
# image = cv2.imread("1.jpg")
# image = cv2.imread("2.jpg")
# image = cv2.imread("3.jpg")
vc = cv2.VideoCapture(0)
while True:
    ret, image = vc.read()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags =cv2.CASCADE_SCALE_IMAGE
    )

    faces_images = []
    print len(faces)
    for (x, y, w, h) in faces:
        crop_face = image[y:y+h,x:x+w]
        gray_face = cv2.cvtColor(crop_face, cv2.COLOR_BGR2GRAY)
        eyes = eyeCascade.detectMultiScale(gray_face)
        for (ex,ey,ew,eh) in eyes:
            print   (ex,ey,ew,eh)
            cv2.rectangle(crop_face,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

        smile = smileCascade.detectMultiScale(gray_face)
        for (ex,ey,ew,eh) in smile:
            print   (ex,ey,ew,eh)
            cv2.rectangle(crop_face,(ex,ey),(ex+ew,ey+eh),(255,0,0),2)

        cv2.imshow("cropface" ,crop_face)
        cv2.waitKey(10)
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)



    #image = cv2.resize(image,None,fx=.5, fy=0.5, interpolation = cv2.INTER_CUBIC)
    cv2.imshow("Faces found" ,image)
    cv2.waitKey(10)