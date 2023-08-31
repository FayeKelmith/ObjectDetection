from transformers import DetrImageProcessor, DetrForObjectDetection 
import torch 
from PIL import Image 
import app
import random
import cv2 
from PIL import Image




pic = cv2.imread('resources/photos/street.jpg')
pic = cv2.resize(pic, (0, 0), fx = 0.2, fy = 0.2)

#INFO: passing in images to process them in model.
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
            cv2.rectangle(pic,start,end,col,thickness=2)
        
        cv2.putText(pic,model.config.id2label[label.item()],(int(box[0]),int(box[3])-5),cv2.FONT_HERSHEY_COMPLEX,fontScale=0.4,color=(255,255,255),thickness=1)
        # print(f'Detected {model.config.id2label[label.item()]} with confidence:' f'{round(score.item(),3)} at location {box}')
    
    cv2.imshow('Street',pic)
    cv2.waitKey(0)

#INFO: to process videos  
      


detection(pic)

#TODO: 
# 1. Create algorithm to generate different colors for particular category of objects
# 1.1 Position the boxes better than they are now, same for the text
# 4. Connect it with streamlit
# 5. Try deployment again
        