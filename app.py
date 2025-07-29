import streamlit as st
import numpy as np
import joblib

# Load model and scaler
model = joblib.load('diabetes_model.pkl')
scaler = joblib.load('scaler.pkl')

# Streamlit UI
st.title("🩺 Diabetes Risk Prediction App")
st.write("This app predicts whether a person is at risk of diabetes based on medical details.")

# Input fields for user data
pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
glucose = st.number_input("Glucose Level", min_value=0, max_value=200, value=100)
bp = st.number_input("Blood Pressure", min_value=0, max_value=150, value=70)
skin = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
insulin = st.number_input("Insulin Level", min_value=0, max_value=900, value=80)
bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5)
age = st.number_input("Age", min_value=0, max_value=120, value=30)

# Prediction button
if st.button("Predict"):
    # Prepare user input
    user_data = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])
    user_data_scaled = scaler.transform(user_data)
    
    # Make prediction
    prediction = model.predict(user_data_scaled)

    # Show result
    if prediction[0] == 1:
        st.error("⚠️ High Risk of Diabetes")
    else:
        st.success("✅ Low Risk of Diabetes")
