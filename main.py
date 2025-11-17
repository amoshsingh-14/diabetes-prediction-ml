from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np

# Load the model - it contains 3 objects: model, scaler, and label_encoder
with open("best_knn_model.pkl", "rb") as f:
    saved_data = pickle.load(f)
    
# Unpack based on what's in the file
if isinstance(saved_data, tuple) and len(saved_data) == 3:
    knn_model, scaler, label_encoder = saved_data
elif isinstance(saved_data, tuple) and len(saved_data) == 2:
    knn_model, scaler = saved_data
    label_encoder = None
else:
    knn_model = saved_data
    scaler = None
    label_encoder = None

app = FastAPI(title="Diabetes Prediction API")

class PatientData(BaseModel):
    age: int
    gender: str  # "Male" or "Female"
    bmi: float
    hypertension: int  # 0 or 1
    heart_disease: int  # 0 or 1
    smoking_history: str  # e.g., "never", "current", "former", "not current", "ever"
    HbA1c_level: float
    blood_glucose_level: float

@app.get("/")
def home():
    return {"message": "Diabetes Prediction API is running"}

@app.post("/predict")
def predict(data: PatientData):
    # Convert gender to numeric (0 for Female, 1 for Male)
    gender_numeric = 1 if data.gender.lower() == "male" else 0
    
    # Convert smoking history to numeric
    smoking_map = {
        "never": 0,
        "no info": 1,
        "current": 2,
        "former": 3,
        "ever": 4,
        "not current": 5
    }
    smoking_numeric = smoking_map.get(data.smoking_history.lower(), 0)
    
    # Create feature array
    features = np.array([[
        data.age, gender_numeric, data.bmi, data.hypertension,
        data.heart_disease, smoking_numeric, data.HbA1c_level, data.blood_glucose_level
    ]])
    
    # Scale features if scaler exists
    if scaler is not None:
        features_scaled = scaler.transform(features)
    else:
        features_scaled = features
    
    # Make prediction
    prediction = knn_model.predict(features_scaled)[0]
    result = "Diabetic" if prediction == 1 else "Not Diabetic"
    
    return {"prediction": result}