import streamlit as st 
import main
import numpy as np
import cv2
from PIL import Image
import time

st.title("Object Detection using DETR model with ResNet-50 backbone trained on the Coco 2017 dataset")

st.divider()

file = st.file_uploader("Please upload an image",accept_multiple_files=False)


# extension = file.name.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'))
if file is not None:
    pil_image = Image.open(file)
    pic_ = np.array(pil_image)
    pic = cv2.resize(pic_, (0, 0), fx = 0.2, fy = 0.2)
    with st.spinner('Wait for it...'):
        st.image(main.detection(pic))
    st.success('There you go!')
    
