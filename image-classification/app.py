import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os

# Load model
model = load_model("model/mobilenet_model.keras")

st.title("Image Classification Demo")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB").resize((224, 224))
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img_array = np.expand_dims(np.array(image) / 255.0, axis=0)

    # Predict
    prediction = model.predict(img_array)
    st.write("Prediction:", prediction.tolist())
