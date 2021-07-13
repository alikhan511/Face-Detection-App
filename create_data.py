# import packages
import cv2, numpy, os, sys

haar_file = 'haarcascade_frontalface_default.xml'
datasets = 'datasets' #All the faces data will be present in this folder
sub_data = input('Enter name of person: ') #these are sub data sets

#making folder of person's images
path = os.path.join(datasets, sub_data)
if not os.path.isdir(path):
    os.mkdir(path)

# size of picture which will be taken
(width, height) = (130, 100)

# sending haar_file to cascadeClassifier class as an object which is in opencv package which implement the camera
face_cascade = cv2.CascadeClassifier(haar_file)
webcamera = cv2.VideoCapture(0) # this class capture the video .... '0' for my webcam if another webcame then we'll use '1' instead of '0'

 # this loop will work until take 100 pictures
count = 1
while count < 101:
    (_, im) = webcamera.read()
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 4)
    for (x, y, w, h) in faces:
        cv2.rectangle(im, (x, y), (x+w, y+h), (255, 0, 0), 2)
        face = gray[y: y+h, x: x+w]
        face_resize = cv2.resize(face, (width, height))
        cv2.imwrite('%s/%s.png' %(path, count), face_resize)
    count += 1
    cv2.imshow('opencv', im)
    cv2.waitKey(10)




