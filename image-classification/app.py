from tensorflow.keras.models import load_model
import streamlit as st
from PIL import Image
import numpy as np

# ✅ Load model using correct filename
model = load_model("model/mobilenet_model.h5")

st.title("Image Classification Demo")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).resize((224, 224))
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img_array = np.expand_dims(np.array(image) / 255.0, axis=0)

    # Predict
    prediction = model.predict(img_array)
    st.write("Prediction:", prediction)

