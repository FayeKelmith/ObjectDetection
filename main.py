from transformers import DetrImageProcessor, DetrForObjectDetection 
import torch 
from PIL import Image 
import app
import cv2 
from PIL import Image



#TODO: I am going to make it work for video, then abstract the code into a function and set up scenes where the user uploads either a video or an image (and even other files to be handled as an error).


video = cv2.VideoCapture('resources/videos/dog.mp4')
    
while True:
    ret, frame = video.read()
        
        #Checking for the end or if video not read correctly
    if not ret:
        break;
        
        #to gray
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    #getting processor and model
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
    
    #FIXME: try converting this PILimage color channel to rgb and run again
    #processing the inputs
    inputs = processor(images=frame,return_tensors='pt')
    outputs= model(**inputs)
    
    #converting outputs back to useful output
    #obtaining original sizes
    target_sizes = torch.tensor([frame.size[::-1]])
    #processing output, thresholding and fitting setting target sizes
    results = processor.post_process_object_detection(outputs,target_sizes=target_sizes,threshhold=0.9)[0]
    
    for score,label,box in zip(results["scores"], results["labels"],results["boxes"]):
        box = [round(i,2) for i in box.tolist()]
        print(f'Detected {model.config.id2label[label.item()]} with confidence:' f'{round(score.item(),3)} at location {box}')
        
    #cv2.imshow('Dog',gray)
    
    if(cv2.waitKey(20) & 0xFF==ord('q')):
        break;
    
#FIXME: this can not work. Opencv is mean to edit the picture and add shapes to the objects. I have to detect objects in each frame, pass them to the model for pediction. Use ouput from model to draw shape on them using opencv. Then at the end display the model as output video format. 

#NOTE: I think it's best if I do it purely offline before running it online. That is use purely open CV. once it works offline, I ship everything to the net using streamlit.
        
        
        
        