import cv2
import numpy as np
import pickle

#using cascade classifier
face_classifier = cv2.CascadeClassifier('D:/Documents/PycharmProjects/face/cascades/data/haarcascade_frontalface_default.xml')#face classifier of cascade

#starting video feed
cap = cv2.VideoCapture(0)
count = 0

while True :
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)#color conversion
    faces = face_classifier.detectMultiScale(gray, 1.5, 5)#detecting faces
    #cropping and saving
    for (x,y,w,h) in faces:
        cropped_face = frame[y:y+h, x:x+w]#cropped face from the video feed
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)#rectangle around face
        count += 1#picture count and naming pictures
        face = cv2.cvtColor(cropped_face,cv2.COLOR_BGR2GRAY)#grayscaling cropped image
        file_name_path = 'D:/Documents/PycharmProjects/face/images/sid/'+str(count)+'.jpg'#path to save image different folder for different person
        cv2.imwrite(file_name_path,face)#saving image
        cv2.putText(frame,str(count),(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2) #mentioning image count on picture
        cv2.imshow('face cropper',frame)#displaying images that are to be saved/live video feed

    if cv2.waitKey(1) == 27 or count == 500: #or count == 15:#used to stop at mentioned no. of images.#press escape button to exit
        break
cap.release()#releasing camera
cv2.destroyAllWindows()#close all cv2 related windows
print('collection completed')