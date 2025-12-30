import streamlit as st
import pickle
import numpy as np

# Load trained model and scaler
with open("logistic_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

st.title("🚢 Titanic Survival Prediction App")

st.write("Enter passenger details (whole numbers only):")

# Input fields with real feature names
Pclass = st.number_input("Passenger Class (1 = 1st, 2 = 2nd, 3 = 3rd)", min_value=1, max_value=3, step=1)
Sex = st.number_input("Sex (0 = Female, 1 = Male)", min_value=0, max_value=1, step=1)
Age = st.number_input("Age (years)", min_value=0, max_value=100, step=1)
SibSp = st.number_input("Number of Siblings/Spouses Aboard", min_value=0, max_value=10, step=1)
Parch = st.number_input("Number of Parents/Children Aboard", min_value=0, max_value=10, step=1)
Fare = st.number_input("Fare Paid", min_value=0, max_value=1000, step=1)
Embarked = st.number_input("Port of Embarkation (0 = C, 1 = Q, 2 = S)", min_value=0, max_value=2, step=1)

if st.button("Predict Survival"):
    # Create input array
    input_data = np.array([[Pclass, Sex, Age, SibSp, Parch, Fare, Embarked]])

    # Scale input
    input_scaled = scaler.transform(input_data)

    # Prediction
    prediction = model.predict(input_scaled)[0]

    if prediction == 1:
        st.success("✅ The passenger is likely to SURVIVE.")
    else:
        st.error("❌ The passenger is NOT likely to survive.")
