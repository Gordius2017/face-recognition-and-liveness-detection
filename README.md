# face-recognition-and-liveness-detection

This project contains the python-flask server that receives the images from the c++ program and processes them for facial recognition. 

The c++ program receives the images from the Raspberry Pi and processes them for liveness detection. In order to use it, the OpenFace library must be downloaded and FaceLandMarkVid.cpp must be replaced in OpenFace/exe/FaceLandMarkVid, and the compiled using the install.sh found in OpenFace/.

In helper-scripts there are many other usefull scripts for face analyzing and communication: blink detection, landmarks detection, sending images through sockets or POST-ing them as base64.
