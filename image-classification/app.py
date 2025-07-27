import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from PIL import Image

# Load model
model = tf.keras.models.load_model('models/mobilenet_model.keras')

# Class names
class_names = ['Cat', 'Dog']  # Change as per your dataset

st.title("Image Classifier")
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image', use_column_width=True)
    
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    st.write(f"Prediction: **{predicted_class}**")


