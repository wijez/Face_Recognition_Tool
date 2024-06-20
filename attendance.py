import face_recognition
import cv2
import os

path = "training"
images = []
classNames = []
mylist = os.listdir(path)

for img in mylist:
    curImg = cv2.imread(f'{path}/{img}')
    images.append(curImg)
    classNames.append(os.path.splitext(img)[0])

