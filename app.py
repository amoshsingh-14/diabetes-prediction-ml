import streamlit as st
import requests

st.set_page_config(page_title="Diabetes Prediction App", layout="centered")
st.title("🩺 Diabetes Prediction App")

# Create two columns for horizontal layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("Personal Information")
    age = st.number_input("Age", min_value=1, max_value=120, value=30)
    gender = st.selectbox("Gender", ["Male", "Female"])
    bmi = st.number_input("BMI (Body Mass Index)", min_value=10.0, max_value=70.0, value=25.0)
    smoking_history = st.selectbox("Smoking History", ["never", "current", "former", "ever", "not current", "No Info"])

with col2:
    st.subheader("Medical Information")
    hypertension = st.selectbox("Hypertension", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    heart_disease = st.selectbox("Heart Disease", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    hba1c = st.number_input("HbA1c Level (%)", min_value=3.0, max_value=15.0, value=5.5, step=0.1)
    blood_glucose = st.number_input("Blood Glucose Level (mg/dL)", min_value=50.0, max_value=400.0, value=100.0)

# Button centered below the columns
st.markdown("---")
if st.button("Predict", use_container_width=True):
    url = "https://diabetes-prediction-ml-x756.onrender.com/predict"
    data = {
        "age": age,
        "gender": gender,
        "bmi": bmi,
        "hypertension": hypertension,
        "heart_disease": heart_disease,
        "smoking_history": smoking_history,
        "HbA1c_level": hba1c,
        "blood_glucose_level": blood_glucose
    }
    
    try:
        response = requests.post(url, json=data)
        result = response.json()
        st.success(f"✅ Prediction: {result['prediction']}")
    except Exception as e:
        st.error(f"Error: {str(e)}")
