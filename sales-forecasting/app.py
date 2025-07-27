import streamlit as st
import pandas as pd
import joblib

st.title("🛒 Sales Forecasting App")

df = pd.read_csv("data/sales_data.csv")
st.write("### Sample Data", df.head())

model = joblib.load("model/sales_model.pkl")
future_days = st.slider("Predict sales for how many future days?", 1, 30, 7)

# Dummy forecast for UI
forecast = model.predict(df.tail(future_days).drop("Sales", axis=1))

st.write(f"### Predicted Sales for Next {future_days} Days")
st.write(forecast)

