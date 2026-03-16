import warnings
import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os

app = FastAPI(title="Heart Failure Prediction API")

# Load model explicitly with joblib using absolute path just in case
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "rf_heart_failure_model.pkl")

# Ignore scikit-learn warnings about version inconsistencies when loading
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    model = joblib.load(MODEL_PATH)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # For development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PatientData(BaseModel):
    age: int
    gender: str
    restingBP: int
    cholesterol: int
    fastingBS: int
    chestPainType: str
    restingECG: str
    maxHR: int
    exerciseAngina: str
    oldpeak: float
    stSlope: str

@app.get("/")
def read_root():
    return {"status": "Heart Failure Prediction API is running with ML model loaded."}

@app.post("/predict")
def predict_risk(data: PatientData):
    # Mapping ST_Slope based on model's training encoding
    # Where Upsloping = 0, Flat = 1, Downsloping = 2
    st_mapping = {"Down": 2, "Flat": 1, "Up": 0, "Downsloping": 2, "Upsloping": 0}
    st_val = st_mapping.get(data.stSlope, 1)

    features = {
        'ST_Slope': st_val,
        'Sex_F': 1 if data.gender == 'F' else 0,
        'Sex_M': 1 if data.gender == 'M' else 0,
        'ChestPainType_ASY': 1 if data.chestPainType == 'ASY' else 0,
        'ChestPainType_ATA': 1 if data.chestPainType == 'ATA' else 0,
        'ChestPainType_NAP': 1 if data.chestPainType == 'NAP' else 0,
        'ChestPainType_TA': 1 if data.chestPainType == 'TA' else 0,
        'RestingECG_LVH': 1 if data.restingECG == 'LVH' else 0,
        'RestingECG_Normal': 1 if data.restingECG == 'Normal' else 0,
        'RestingECG_ST': 1 if data.restingECG == 'ST' else 0,
        'ExerciseAngina_N': 1 if data.exerciseAngina == 'N' else 0,
        'ExerciseAngina_Y': 1 if data.exerciseAngina == 'Y' else 0,
        'Age': data.age,
        'RestingBP': data.restingBP,
        'Cholesterol': data.cholesterol,
        'MaxHR': data.maxHR,
        'Oldpeak': data.oldpeak,
        'FastingBS': data.fastingBS
    }
    
    # Define exact order model expects
    feature_order = [
        'ST_Slope', 'Sex_F', 'Sex_M', 'ChestPainType_ASY', 'ChestPainType_ATA', 
        'ChestPainType_NAP', 'ChestPainType_TA', 'RestingECG_LVH', 
        'RestingECG_Normal', 'RestingECG_ST', 'ExerciseAngina_N', 
        'ExerciseAngina_Y', 'Age', 'RestingBP', 'Cholesterol', 
        'MaxHR', 'Oldpeak', 'FastingBS'
    ]
    
    # Create single row DataFrame
    df = pd.DataFrame([{col: features[col] for col in feature_order}])
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prediction = model.predict(df)[0]
        prob = model.predict_proba(df)[0]
    
    # Using class index 1 (positive for heart failure) for probability
    prob_percent = int(prob[1] * 100)
    
    riskLevel = "High Risk" if prediction == 1 else "Low Risk"
    
    # Rhythm mapping based on RestingECG since prediction represents failure overall
    if data.restingECG == 'Normal':
        rhythm = "Normal Sinus Rhythm"
    elif data.restingECG == 'LVH':
        rhythm = "Left Ventricular Hypertrophy"
    else:
        rhythm = "ST-T Wave Abnormality"
        
    return {
        "riskLevel": riskLevel,
        "probability": prob_percent,
        "rhythm": rhythm
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
