import streamlit as st 

st.title("Object Detection using DETR model with ResNet-50 backbone trained on the Coco 2017 dataset")

st.header('You can upload a video or stream one')
st.divider()

file = st.file_uploader("Please upload a video less than 200MB",accept_multiple_files=False)

#Note: image identifier
#filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'))

#TODO: ship this video now to opencv for processin
