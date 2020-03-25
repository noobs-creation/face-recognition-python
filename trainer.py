import cv2
import numpy as np
import os
import pickle
from PIL import Image

base_dir = os.path.dirname(os.path.abspath(__file__)) #locating path to this file
image_dir = os.path.join(base_dir,"images")#path to images folder
face_cascade = cv2.CascadeClassifier('D:/Documents/PycharmProjects/face/cascades/data/haarcascade_frontalface_default.xml') #haarcascade classifier

recognizer = cv2.face.LBPHFaceRecognizer_create()   #trainer model used for starting recognition

current_id=0
label_ids={}
y_labels=[]
x_train=[]

for root,dirs,files in os.walk(image_dir):#working in the image directory
    for file in files:
        if file.endswith("png") or file.endswith("jpg"):
            path = os.path.join(root,file)
            label = os.path.basename(root).replace(" ","-").lower()
            if not label in label_ids:
                label_ids[label] = current_id
                current_id += 1
            id_ = label_ids[label]
            pil_image = Image.open(path).convert("L")#grayscale image conversion
            final_image = pil_image.resize((550, 550), Image.ANTIALIAS)#final image resized
            image_array = np.array(final_image,"uint8")
            faces = face_cascade.detectMultiScale(image_array,1.5,5)#detecting faces
            for (x,y,w,h) in faces:
                roi = image_array[y:y+h, x:x+w]
                x_train.append(roi)
                y_labels.append(id_)


with open("pickles/face_labels.pickle",'wb') as f:#saving pickle wb=write byte
    pickle.dump(label_ids,f)

recognizer.train(x_train,np.asarray(y_labels))#training the model
recognizer.save("recognizers/face-trainning.yml")#saving train model

print ("trainning complete")