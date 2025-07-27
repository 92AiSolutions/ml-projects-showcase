import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

st.title("📷 Image Classification")

model = tf.keras.models.load_model("model/mobilenet_model.h5")

uploaded = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded:
    img = Image.open(uploaded).resize((224, 224))
    st.image(img)
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)
    prediction = model.predict(img_array)
    st.write("Predicted Class:", "Cat" if prediction[0][0] < 0.5 else "Dog")

