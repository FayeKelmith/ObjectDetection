# from transformers import DetrImageProcessor, DetrForObjectDetection 
# import torch 
# from PIL import Image 
import app
import cv2 
from PIL import Image



#TODO: I am going to make it work for video, then abstract the code into a function and set up scenes where the user uploads either a video or an image (and even other files to be handled as an error).

    
video = cv2.VideoCapture(app.vid)
    
while True:
    ret, frame = video.read()
        
        #Checking for the end or if video not read correctly
    if not ret:
        break;
        
        #to gray
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#FIXME: this can not work. Opencv is mean to edit the picture and add shapes to the objects. I have to detect objects in each frame, pass them to the model for pediction. Use ouput from model to draw shape on them using opencv. Then at the end display the model as output video format. 

#NOTE: I think it's best if I do it purely offline before running it online. That is use purely open CV. once it works offline, I ship everything to the net using streamlit.
        
        
        
        