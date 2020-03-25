import cv2
import numpy as np
import pickle

face_classifier = cv2.CascadeClassifier('D:/Documents/PycharmProjects/face/cascades/data/haarcascade_frontalface_default.xml')#face classifier of cascade

recognizer = cv2.face.LBPHFaceRecognizer_create()   #trainer model used for starting recognition
recognizer.read("./recognizers/face-trainning.yml") #reading the trained recognizer

labels={"person_name":1}
with open("pickles/face_labels.pickle",'rb') as f:#reading the pickle rb=read byte
    og_labels = pickle.load(f)
    labels = {v:k for k,v in og_labels.items()}#person name and numbers were loaded reversing it

cap = cv2.VideoCapture(0)
while True :
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)#color to gray
    faces = face_classifier.detectMultiScale(gray,1.35,5)#detecting face
    for (x,y,w,h) in faces: #getting faces
        roi_gray = gray[y:y+h,x:x+w]
        roi_color = frame[y:y + h, x:x + w]
        id_,conf = recognizer.predict(roi_gray)#recognising face with trained model
        if conf>=65:
            font=cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id_]#naming the face if the confidence level meets the threshold
            color = (0,255,255)
            stroke =2
            cv2.putText(frame,name,(x,y),font,1,color,stroke,cv2.LINE_AA)
        #cv2.imwrite("path name and file name ",roi_color)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,255),2)
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) == 27:  #press escape button to exit
        break

cap.release()
cv2.destroyAllWindows()