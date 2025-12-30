import streamlit as st
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("logistic_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("🚢 Titanic Survival Prediction")

st.write("Enter passenger details:")

# ---- Inputs ----
Pclass = st.selectbox("Passenger Class", [1, 2, 3])

Sex = st.selectbox("Sex", ["Female", "Male"])
Sex = 1 if Sex == "Male" else 0

Age = st.number_input(
    "Age (years)",
    min_value=0,
    max_value=100,
    value=30,
    step=1   # whole numbers only ✅
)

Fare = st.number_input(
    "Fare",
    min_value=0.0,
    max_value=600.0,
    value=32.0
)

IsAlone = st.selectbox(
    "Is Passenger Alone?",
    ["Yes", "No"]
)
IsAlone = 1 if IsAlone == "Yes" else 0

# ---- Prediction ----
if st.button("Predict Survival"):
    input_data = np.array([[Pclass, Sex, Age, Fare, IsAlone]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]

    if prediction == 1:
        st.success("✅ Passenger is likely to SURVIVE")
    else:
        st.error("❌ Passenger is NOT likely to survive")
