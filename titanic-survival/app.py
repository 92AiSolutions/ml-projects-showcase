import streamlit as st
import joblib
import pandas as pd

st.title("🚢 Titanic Survival Predictor")

model = joblib.load("model/titanic_model.pkl")

age = st.slider("Age", 1, 80, 25)
fare = st.slider("Fare", 0, 500, 50)
pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
embarked = st.selectbox("Port of Embarkation", ["C", "Q", "S"])

input_data = pd.DataFrame({
    "Age": [age],
    "Fare": [fare],
    "Pclass": [pclass],
    "Sex": [1 if sex == "male" else 0],
    "Embarked": [{"C": 0, "Q": 1, "S": 2}[embarked]]
})

if st.button("Predict"):
    result = model.predict(input_data)
    st.write("Survival:", "✅ Survived" if result[0] == 1 else "❌ Did not survive")

