import warnings
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os

app = FastAPI(title="Heart Failure Prediction API")

# ── Load all model artifacts ──────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

with warnings.catch_warnings():
    warnings.simplefilter("ignore")

    # 1. RFE-selected feature names (list of 10 feature names)
    selected_features = joblib.load(
        os.path.join(BASE_DIR, "selected_features_rfe_names.pkl")
    )

    # 2. Preprocessor (ColumnTransformer – may fail on sklearn version mismatch)
    try:
        preprocessor = joblib.load(
            os.path.join(BASE_DIR, "preprocessor.pkl")
        )
    except Exception:
        preprocessor = None  # Fall back to manual encoding

    # 3. RFE selector (fitted sklearn RFE object)
    rfe_selector = joblib.load(
        os.path.join(BASE_DIR, "rfe_selector.pkl")
    )

    # 4. Random Forest model trained on RFE-selected features
    model = joblib.load(
        os.path.join(BASE_DIR, "rf_rfe_heart_failure_model.pkl")
    )

# ── CORS ──────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request Schema (unchanged) ────────────────────────────────────────
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

# ── Helpers ───────────────────────────────────────────────────────────

# Full set of one-hot encoded columns (matches the preprocessor output order)
ALL_ENCODED_COLUMNS = [
    'ST_Slope', 'Sex_F', 'Sex_M',
    'ChestPainType_ASY', 'ChestPainType_ATA',
    'ChestPainType_NAP', 'ChestPainType_TA',
    'RestingECG_LVH', 'RestingECG_Normal', 'RestingECG_ST',
    'ExerciseAngina_N', 'ExerciseAngina_Y',
    'Age', 'RestingBP', 'Cholesterol',
    'MaxHR', 'Oldpeak', 'FastingBS'
]

def _manual_encode(data: PatientData) -> pd.DataFrame:
    """Manually one-hot encode patient data (fallback when preprocessor
    can't be loaded due to sklearn version mismatch)."""
    st_mapping = {"Down": 2, "Flat": 1, "Up": 0, "Downsloping": 2, "Upsloping": 0}

    features = {
        'ST_Slope':           st_mapping.get(data.stSlope, 1),
        'Sex_F':              1 if data.gender == 'F' else 0,
        'Sex_M':              1 if data.gender == 'M' else 0,
        'ChestPainType_ASY':  1 if data.chestPainType == 'ASY' else 0,
        'ChestPainType_ATA':  1 if data.chestPainType == 'ATA' else 0,
        'ChestPainType_NAP':  1 if data.chestPainType == 'NAP' else 0,
        'ChestPainType_TA':   1 if data.chestPainType == 'TA' else 0,
        'RestingECG_LVH':     1 if data.restingECG == 'LVH' else 0,
        'RestingECG_Normal':  1 if data.restingECG == 'Normal' else 0,
        'RestingECG_ST':      1 if data.restingECG == 'ST' else 0,
        'ExerciseAngina_N':   1 if data.exerciseAngina == 'N' else 0,
        'ExerciseAngina_Y':   1 if data.exerciseAngina == 'Y' else 0,
        'Age':                data.age,
        'RestingBP':          data.restingBP,
        'Cholesterol':        data.cholesterol,
        'MaxHR':              data.maxHR,
        'Oldpeak':            data.oldpeak,
        'FastingBS':          data.fastingBS,
    }
    return pd.DataFrame([{col: features[col] for col in ALL_ENCODED_COLUMNS}])


def _preprocess(data: PatientData) -> pd.DataFrame:
    """Run through the saved preprocessor if available, else manual encode."""
    if preprocessor is not None:
        # Build raw DataFrame matching the preprocessor's expected input
        raw = pd.DataFrame([{
            'Age':           data.age,
            'Sex':           data.gender,
            'ChestPainType': data.chestPainType,
            'RestingBP':     data.restingBP,
            'Cholesterol':   data.cholesterol,
            'FastingBS':     data.fastingBS,
            'RestingECG':    data.restingECG,
            'MaxHR':         data.maxHR,
            'ExerciseAngina': data.exerciseAngina,
            'Oldpeak':       data.oldpeak,
            'ST_Slope':      data.stSlope,
        }])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            transformed = preprocessor.transform(raw)
        # The preprocessor may return a sparse matrix or ndarray
        if hasattr(transformed, 'toarray'):
            transformed = transformed.toarray()
        col_names = (preprocessor.get_feature_names_out()
                     if hasattr(preprocessor, 'get_feature_names_out')
                     else ALL_ENCODED_COLUMNS)
        return pd.DataFrame(transformed, columns=col_names)
    else:
        return _manual_encode(data)


def _select_rfe_features(df_encoded: pd.DataFrame) -> pd.DataFrame:
    """Apply RFE feature selection — use the saved feature name list
    for a direct column pick (safest), fall back to the selector object."""
    # Direct column selection using the saved feature names
    missing = [f for f in selected_features if f not in df_encoded.columns]
    if not missing:
        return df_encoded[selected_features]

    # Fallback: use the RFE selector's transform (works on positional order)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        arr = rfe_selector.transform(df_encoded.values)
    return pd.DataFrame(arr, columns=selected_features)


# ── Routes ────────────────────────────────────────────────────────────

@app.get("/")
def read_root():
    return {
        "status": "Heart Failure Prediction API is running with ML model loaded.",
        "model": "Random Forest (RFE-selected features)",
        "features_used": selected_features,
    }


@app.post("/predict")
def predict_risk(data: PatientData):
    # Step 1: Encode raw input → full feature DataFrame
    df_encoded = _preprocess(data)

    # Step 2: Select the 10 RFE-chosen features
    df_selected = _select_rfe_features(df_encoded)

    # Step 3: Predict
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prediction = model.predict(df_selected)[0]
        prob = model.predict_proba(df_selected)[0]

    # Using class index 1 (positive for heart failure) for probability
    prob_percent = int(prob[1] * 100)
    riskLevel = "High Risk" if prediction == 1 else "Low Risk"

    # Rhythm mapping based on RestingECG
    if data.restingECG == 'Normal':
        rhythm = "Normal Sinus Rhythm"
    elif data.restingECG == 'LVH':
        rhythm = "Left Ventricular Hypertrophy"
    else:
        rhythm = "ST-T Wave Abnormality"

    return {
        "riskLevel": riskLevel,
        "probability": prob_percent,
        "rhythm": rhythm,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
