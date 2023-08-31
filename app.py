from transformers import DetrImageProcessor, DetrForObjectDetection 
import torch  
import random
import streamlit as st 
import numpy as np
import cv2
from PIL import Image
import time

st.title("Object Detection using DETR model with ResNet-50 backbone trained on the Coco 2017 dataset")

st.divider()

file = st.file_uploader("Please upload an image",accept_multiple_files=False)


def random_color():
        levels = range(32,256,32)
        return tuple(random.choice(levels) for _ in range(3))

def detection(pic):
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
    
    #converting the opencv image back to pil type that is compatible with the model.
    pil_image = Image.fromarray(cv2.cvtColor(pic, cv2.COLOR_BGR2RGB))
    
    inputs = processor(images=pil_image,return_tensors='pt')
    outputs= model(**inputs)
    
    #converting outputs back to useful output
    #obtaining original sizes
    target_sizes = torch.tensor([pil_image.size[::-1]])
    #processing output, thresholding and fitting setting target sizes
    results = processor.post_process_object_detection(outputs,target_sizes=target_sizes,threshold=0.9)[0]
    
    #List to store differentt colors in them
    colors = {}
    
    
    for score,label,box in zip(results["scores"], results["labels"],results["boxes"]):
        box = [round(i,2) for i in box.tolist()]
        start = (int(box[0]),int(box[1]))
        end = (int(box[2]),int(box[3]))
        
        if label not in colors:
            col =  random_color()
            colors[label] = col
            #to obtain dimensions of detected image
        
        cv2.rectangle(pic,start,end,colors[label],thickness=2)
        cv2.putText(pic,model.config.id2label[label.item()],(int(box[0]),int(box[3])-5),cv2.FONT_HERSHEY_COMPLEX,fontScale=0.4,color=(205,205,205),thickness=1)
        # print(f'Detected {model.config.id2label[label.item()]} with confidence:' f'{round(score.item(),3)} at location {box}')
        
    return pic
if file is not None:
    pil_image = Image.open(file)
    pic_ = np.array(pil_image)
    pic = cv2.resize(pic_, (0, 0), fx = 0.2, fy = 0.2)
    with st.spinner('Wait for it...'):
        st.image(detection(pic))
    st.success('There you go!')
    
