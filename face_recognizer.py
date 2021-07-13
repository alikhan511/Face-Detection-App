# import packages
import cv2, numpy, os

haar_file = 'haarcascade_frontalface_default.xml'
datasets = 'datasets' #All the faces data will be taken from this folder

# part1 we'll train the model by using pictures which are available in subsets
# create a list of images and corresponding names
images = []
labels = []
names = {}
id = 0

for(subdirs, dirs, files) in os.walk(datasets):
    for subdir in dirs:
        names[id] = subdir
        pics_path = os.path.join(datasets, subdir)
        for filename in os.listdir(pics_path):
            path = pics_path + '/' + filename
            label = id
            images.append(cv2.imread(path, 0))
            labels.append(int(label))
        id += 1

(width, height) = (130, 100)
# create a numpy array from lists above
(images, labels) = [numpy.array(list1) for list1 in [images, labels]]

# openCV trains the model from images

model = cv2.face.LBPHFaceRecognizer_create()

model.train(images, labels)

# part 2 take picture from camera and recognize the person

face_cascade = cv2.CascadeClassifier(haar_file)

webcamera = cv2.VideoCapture(0)

while True:
    (_, im) = webcamera.read()
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(im, (x, y), (x+w, y+h), (255, 0, 0), 2)
        face = gray[y: y+h, x: x+w]
        face_resize = cv2.resize(face, (width, height))
        # try to recognize the face
        prediction = model.predict(face_resize)
        cv2.rectangle(im, (x, y), (x+w, y+h), (0, 255 , 0), 2)

        if prediction[1] < 100:
            cv2.putText(im, '%s - %.0f' %(names[prediction[0]], prediction[1]), (x - 10, y - 10),
                        cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
        else:
            cv2.putText(im, 'not recognized', (x - 10, y - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))

    cv2.imshow('openCV', im)
    cv2.waitKey(10)