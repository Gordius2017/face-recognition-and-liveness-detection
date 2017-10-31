import sys
import dlib
from skimage import io

# Take the image file name from the command line
file_name = 'download.jpeg'
file_name = 'crowd_2.jpg'
file_name = '1.jpg'
file_name = '2.jpg'
file_name = '3.jpg'

# Create a HOG face detector using the built-in dlib class
face_detector = dlib.get_frontal_face_detector()

win = dlib.image_window()

# Load the image into an array
image = io.imread(file_name)

# Run the HOG face detector on the image data.
# The result will be the bounding boxes of the faces in our image.
detected_faces = face_detector(image, 1)

print("I found {} faces in the file {}".format(len(detected_faces), file_name))

# Open a window on the desktop showing the image
win.set_image(image)

# Loop through each face we found in the image
print len(detected_faces)
for i, face_rect in enumerate(detected_faces):

    # Detected faces are returned as an object with the coordinates
    # of the top, left, right and bottom edges
    print("- Face #{} found at Left: {} Top: {} Right: {} Bottom: {}".format(i, face_rect.left(), face_rect.top(), face_rect.right(), face_rect.bottom()))

    # Draw a box around each face we found
    win.add_overlay(face_rect)

import cv2
# Wait until the user hits <enter> to close the window
cv2.imshow("image",image)
cv2.waitKey(1000)
dlib.hit_enter_to_continue()
