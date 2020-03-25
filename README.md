# face-recognition-python

<p>
A simple python project for recognition of faces. There are 3 python files, where face_capture.py captures and stores by default 500 pictures from webcam of your pc, next comes, trainer.py which uses yolo and the captured images to train the model for recognising the captured face, next comes face_recog.py which streams live feed from the webcam and recognises the available face.</p>
<p>Being in the learning stage while making this project the frame rate while recognising the face is only 1-2 fps.</p>

<h3>
Changes required to run successfully on your pc :
</h3>

<p>
<i>face_capture.py</i>
<li>Change the destination of Face-Classifier to the downloaded location of this git, in line 6.</li>
<li>Change the location of folder for storing your captured images in line 22, and change folder name to your name replacing 'sid' named folder.</li>

</p>

<p>
<i>trainer.py</i>
<li>Change the destination of Face-Classifier to the downloaded location of this git, in line 9.</li>
</p>

<p>
<i>face_recog.py</i>
<li>Change the destination of Face-Classifier to the downloaded location of this git, in line 5.</li>
</p>
