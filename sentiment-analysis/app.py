import streamlit as st
import joblib

st.title("🔍 Sentiment Analysis")

model = joblib.load("model/sentiment_model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")

text = st.text_area("Enter a review:")

if st.button("Analyze"):
    vec = vectorizer.transform([text])
    result = model.predict(vec)
    st.write("Sentiment:", "✅ Positive" if result[0] == 1 else "❌ Negative")

