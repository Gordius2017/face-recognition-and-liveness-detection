import face_recognition

image = face_recognition.load_image_file("/home/alex/Desktop/unknown_face/21png.png")
face_locations = face_recognition.face_locations(image)
print face_locations
face_landmarks_list = face_recognition.face_landmarks(image)
print face_landmarks_list


picture_of_me = image
my_face_encoding = face_recognition.face_encodings(picture_of_me)[0]

unknown_picture = face_recognition.load_image_file("/home/alex/Desktop/unknown_face/13.png")
unknown_face_encoding = face_recognition.face_encodings(unknown_picture)[0]

results = face_recognition.compare_faces([my_face_encoding], unknown_face_encoding)

if results[0] == True:
    print("It's a picture of me!")
else:
    print("It's not a picture of me!")