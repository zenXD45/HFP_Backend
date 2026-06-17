# Heart Failure Prediction — Backend

A **FastAPI**-powered REST API that receives patient clinical data, runs it through a trained **Random Forest** machine-learning pipeline with **Recursive Feature Elimination (RFE)**, and returns a heart-failure risk prediction.

---

## Table of Contents

1. [Project Structure](#project-structure)
2. [ML Pipeline Overview](#ml-pipeline-overview)
3. [Model Artifacts (.pkl Files)](#model-artifacts-pkl-files)
4. [API Reference](#api-reference)
5. [Code Walkthrough — `main.py`](#code-walkthrough--mainpy)
6. [Test Scripts](#test-scripts)
7. [Configuration Files](#configuration-files)
8. [Local Development](#local-development)
9. [Deployment (Render)](#deployment-render)

---

## Project Structure

```
backend/
├── main.py                           # FastAPI application (core server)
├── preprocessor.pkl                  # Saved sklearn ColumnTransformer
├── rfe_selector.pkl                  # Saved sklearn RFE selector
├── selected_features_rfe_names.pkl   # List of 10 selected feature names
├── rf_rfe_heart_failure_model.pkl    # Trained Random Forest model
├── requirements.txt                  # Python dependencies
├── Procfile                          # Render deployment command
├── test_api.py                       # HTTP-level API test
├── test_function_p.py                # Direct function-level test
├── venv/                             # Python virtual environment
└── README.md                         # This file
```

---

## ML Pipeline Overview

The prediction pipeline follows a **3-stage** process for every incoming request:

```
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  1. PREPROCESS   │ ──▶ │  2. RFE SELECT   │ ──▶ │   3. PREDICT     │
│                  │     │                  │     │                  │
│ Raw patient data │     │ 18 features ──▶  │     │ 10 features ──▶  │
│ ──▶ One-hot      │     │ 10 best features │     │ Risk probability │
│ encoded features │     │                  │     │                  │
└──────────────────┘     └──────────────────┘     └──────────────────┘
   preprocessor.pkl       rfe_selector.pkl +       rf_rfe_heart_
   (with fallback)        selected_features_       failure_model.pkl
                          rfe_names.pkl
```

### Why RFE?

**Recursive Feature Elimination** was applied during model training to identify the **10 most informative features** out of 18 total encoded features. This:
- Reduces overfitting by removing noise features
- Improves prediction accuracy
- Speeds up inference

### The 10 Selected Features

| #  | Feature              | Type     | Description                         |
|----|----------------------|----------|-------------------------------------|
| 1  | `ST_Slope`           | Ordinal  | ST segment slope (Up=0, Flat=1, Down=2) |
| 2  | `Sex_F`              | Binary   | Patient is female                   |
| 3  | `ChestPainType_ASY`  | Binary   | Asymptomatic chest pain             |
| 4  | `ExerciseAngina_N`   | Binary   | No exercise-induced angina          |
| 5  | `ExerciseAngina_Y`   | Binary   | Has exercise-induced angina         |
| 6  | `Age`                | Numeric  | Patient age in years                |
| 7  | `RestingBP`          | Numeric  | Resting blood pressure (mm Hg)      |
| 8  | `Cholesterol`        | Numeric  | Serum cholesterol (mg/dl)           |
| 9  | `MaxHR`              | Numeric  | Maximum heart rate achieved         |
| 10 | `Oldpeak`            | Numeric  | ST depression (exercise vs rest)    |

### The 8 Eliminated Features

`Sex_M`, `ChestPainType_ATA`, `ChestPainType_NAP`, `ChestPainType_TA`, `RestingECG_LVH`, `RestingECG_Normal`, `RestingECG_ST`, `FastingBS` — these were ranked lower by RFE and removed to improve model performance.

---

## Model Artifacts (.pkl Files)

All `.pkl` files are serialized Python objects saved with `joblib`. They are loaded once at server startup and reused for every prediction.

### 1. `preprocessor.pkl` — ColumnTransformer
- **Type:** `sklearn.compose.ColumnTransformer`
- **Size:** ~5 KB
- **Purpose:** Transforms raw patient data (with string categorical columns like `"M"`, `"ASY"`, `"Normal"`) into a fully numeric 18-column DataFrame via one-hot encoding and ordinal mapping.
- **Note:** This artifact may fail to load if the deployment environment has a different `scikit-learn` version than the training environment. The backend has a **graceful fallback** — if loading fails, it uses manual encoding logic that produces identical output.

### 2. `rfe_selector.pkl` — RFE Selector
- **Type:** `sklearn.feature_selection.RFE`
- **Size:** ~4.5 MB
- **Purpose:** A fitted Recursive Feature Elimination object that knows which 10 of the 18 features to keep. It stores:
  - `support_`: Boolean mask of selected features
  - `ranking_`: Importance ranking of all 18 features
  - `feature_names_in_`: The 18 input feature names it expects
- **Usage:** Acts as a secondary fallback for feature selection. The primary method uses the saved feature name list (see below) for a direct column pick.

### 3. `selected_features_rfe_names.pkl` — Feature Name List
- **Type:** Python `list` of 10 strings
- **Size:** ~143 bytes
- **Content:**
  ```python
  ['ST_Slope', 'Sex_F', 'ChestPainType_ASY', 'ExerciseAngina_N',
   'ExerciseAngina_Y', 'Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
  ```
- **Purpose:** The safest and most portable way to select the correct features — just pick these 10 columns by name from the encoded DataFrame. No sklearn version dependency.

### 4. `rf_rfe_heart_failure_model.pkl` — Random Forest Classifier
- **Type:** `sklearn.ensemble.RandomForestClassifier`
- **Size:** ~3.7 MB
- **Purpose:** The trained prediction model. It expects exactly 10 features (the RFE-selected ones) in the correct order and outputs:
  - `predict()` → `0` (No Heart Failure) or `1` (Heart Failure)
  - `predict_proba()` → Probability array, e.g., `[0.20, 0.80]`

---

## API Reference

### `GET /` — Health Check

Verifies the server is running and the model is loaded.

**Response:**
```json
{
  "status": "Heart Failure Prediction API is running with ML model loaded.",
  "model": "Random Forest (RFE-selected features)",
  "features_used": ["ST_Slope", "Sex_F", "ChestPainType_ASY", "ExerciseAngina_N",
                     "ExerciseAngina_Y", "Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"]
}
```

---

### `POST /predict` — Predict Heart Failure Risk

Accepts patient clinical data and returns a risk assessment.

**Request Body:**
```json
{
  "age": 55,
  "gender": "M",
  "restingBP": 120,
  "cholesterol": 198,
  "fastingBS": 0,
  "chestPainType": "ASY",
  "restingECG": "Normal",
  "maxHR": 150,
  "exerciseAngina": "N",
  "oldpeak": 1.5,
  "stSlope": "Flat"
}
```

**Field Definitions:**

| Field           | Type    | Accepted Values                                 |
|-----------------|---------|--------------------------------------------------|
| `age`           | `int`   | Patient age (e.g., `55`)                        |
| `gender`        | `str`   | `"M"` or `"F"`                                  |
| `restingBP`     | `int`   | Resting blood pressure in mm Hg (e.g., `120`)   |
| `cholesterol`   | `int`   | Serum cholesterol in mg/dl (e.g., `198`)         |
| `fastingBS`     | `int`   | Fasting blood sugar > 120 mg/dl: `1` = true, `0` = false |
| `chestPainType` | `str`   | `"ASY"`, `"ATA"`, `"NAP"`, or `"TA"`           |
| `restingECG`    | `str`   | `"Normal"`, `"LVH"`, or `"ST"`                 |
| `maxHR`         | `int`   | Maximum heart rate achieved (e.g., `150`)        |
| `exerciseAngina`| `str`   | `"Y"` or `"N"`                                  |
| `oldpeak`       | `float` | ST depression value (e.g., `1.5`)               |
| `stSlope`       | `str`   | `"Up"`, `"Flat"`, or `"Down"`                   |

**Response:**
```json
{
  "riskLevel": "High Risk",
  "probability": 80,
  "rhythm": "Normal Sinus Rhythm"
}
```

| Response Field  | Type   | Description                                       |
|-----------------|--------|---------------------------------------------------|
| `riskLevel`     | `str`  | `"High Risk"` (prediction = 1) or `"Low Risk"` (prediction = 0) |
| `probability`   | `int`  | Heart failure probability as a percentage (0–100) |
| `rhythm`        | `str`  | ECG rhythm description based on `restingECG` input |

**Rhythm Mapping:**
| `restingECG` Input | Rhythm Output                    |
|---------------------|----------------------------------|
| `"Normal"`          | `"Normal Sinus Rhythm"`          |
| `"LVH"`            | `"Left Ventricular Hypertrophy"` |
| `"ST"`             | `"ST-T Wave Abnormality"`        |

---

## Code Walkthrough — `main.py`

### 1. Imports & App Initialization

```python
import warnings, joblib, numpy as np, pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os

app = FastAPI(title="Heart Failure Prediction API")
```

- **FastAPI** creates the web server application.
- **joblib** loads serialized `.pkl` model artifacts.
- **pandas** constructs DataFrames that match the model's expected input format.
- **warnings** suppresses `scikit-learn` version mismatch warnings during artifact loading and prediction.

### 2. Loading Model Artifacts (Startup)

```python
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

with warnings.catch_warnings():
    warnings.simplefilter("ignore")

    selected_features = joblib.load(os.path.join(BASE_DIR, "selected_features_rfe_names.pkl"))

    try:
        preprocessor = joblib.load(os.path.join(BASE_DIR, "preprocessor.pkl"))
    except Exception:
        preprocessor = None  # Graceful fallback

    rfe_selector = joblib.load(os.path.join(BASE_DIR, "rfe_selector.pkl"))
    model = joblib.load(os.path.join(BASE_DIR, "rf_rfe_heart_failure_model.pkl"))
```

All 4 `.pkl` files are loaded **once** at server startup into memory. This avoids re-reading from disk on every request.

The `preprocessor.pkl` load is wrapped in a `try/except` because it contains a `ColumnTransformer` that depends on a specific `scikit-learn` version. If the deployment environment has a different version, it gracefully falls back to manual encoding (see below).

### 3. CORS Middleware

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

The frontend (e.g., `localhost:5173` in development or a Capacitor mobile app) and backend (`localhost:8000` or Render URL) run on different origins. Without CORS middleware, browsers block cross-origin requests for security. This configuration allows requests from **any** origin.

### 4. Request Schema (Pydantic)

```python
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
```

FastAPI uses Pydantic models for **automatic request validation**. If the frontend sends missing or incorrectly typed fields, FastAPI returns a `422 Unprocessable Entity` error *before* the prediction logic runs.

### 5. Preprocessing — `_manual_encode()`

```python
def _manual_encode(data: PatientData) -> pd.DataFrame:
```

This function replicates what the `preprocessor.pkl` ColumnTransformer does, but without any sklearn dependency:

- **Ordinal Encoding** for `ST_Slope`: Converts `"Up"` → `0`, `"Flat"` → `1`, `"Down"` → `2`
- **One-Hot Encoding** for categorical fields: Expands `gender`, `chestPainType`, `restingECG`, and `exerciseAngina` into binary columns (e.g., `gender="M"` → `Sex_M=1, Sex_F=0`)
- **Passthrough** for numeric fields: `Age`, `RestingBP`, `Cholesterol`, `MaxHR`, `Oldpeak`, `FastingBS`

Returns a single-row DataFrame with **18 columns** in the exact order the model pipeline expects.

### 6. Preprocessing — `_preprocess()`

```python
def _preprocess(data: PatientData) -> pd.DataFrame:
```

A wrapper that:
1. **Tries** to use the saved `preprocessor.pkl` (ColumnTransformer) for encoding
2. **Falls back** to `_manual_encode()` if the preprocessor couldn't be loaded

This dual strategy ensures the backend works regardless of the `scikit-learn` version in the deployment environment.

### 7. Feature Selection — `_select_rfe_features()`

```python
def _select_rfe_features(df_encoded: pd.DataFrame) -> pd.DataFrame:
```

Takes the full 18-column encoded DataFrame and narrows it down to **10 columns** using:

1. **Primary method:** Direct column selection using `selected_features_rfe_names.pkl` (just picks columns by name — most portable, zero sklearn dependency)
2. **Fallback:** Uses the `rfe_selector.pkl` RFE object's `.transform()` method (position-based selection)

### 8. Prediction — `POST /predict`

```python
@app.post("/predict")
def predict_risk(data: PatientData):
    df_encoded = _preprocess(data)           # Step 1: Encode
    df_selected = _select_rfe_features(df_encoded)  # Step 2: Select features
    prediction = model.predict(df_selected)[0]       # Step 3: Predict
    prob = model.predict_proba(df_selected)[0]
```

The prediction endpoint orchestrates the full pipeline:
1. **Encode** raw patient data into 18 numeric features
2. **Select** the 10 RFE-chosen features
3. **Predict** using the Random Forest model
4. **Format** the output into the JSON response (`riskLevel`, `probability`, `rhythm`)

### 9. Server Entry Point

```python
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

When running `python main.py` directly, Uvicorn starts the ASGI server on port `8000`, accessible from any network interface (`0.0.0.0`).

---

## Test Scripts

### `test_function_p.py` — Direct Function Test

```python
from main import app, PatientData, predict_risk
data = PatientData(**{...})
print(predict_risk(data))
```

Tests the `predict_risk` function directly without HTTP. Useful for debugging the ML pipeline in isolation.

**Run:** `python test_function_p.py`

### `test_api.py` — HTTP API Test

Contains a sample patient payload for testing the API via HTTP requests.

**Run:** Start the server first, then run `python test_api.py`

---

## Configuration Files

### `requirements.txt`

```
fastapi
uvicorn
pandas
scikit-learn==1.6.1
pydantic
starlette
joblib
```

All Python dependencies. The `scikit-learn` version is pinned to `1.6.1` to match the version used during model training — this minimizes deserialization issues with the `.pkl` files.

### `Procfile`

```
web: uvicorn main:app --host 0.0.0.0 --port ${PORT:-10000}
```

Used by **Render** (and other PaaS platforms like Heroku) to start the application. It runs Uvicorn with the port provided by the platform's `PORT` environment variable, defaulting to `10000`.

---

## Local Development

### Prerequisites

- Python 3.10+
- pip

### Setup

```bash
# 1. Navigate to the backend directory
cd backend

# 2. Create virtual environment
python -m venv venv

# 3. Activate virtual environment
source venv/bin/activate        # Linux/macOS
# venv\Scripts\activate         # Windows

# 4. Install dependencies
pip install -r requirements.txt
```

### Run the Server

```bash
# Option A: Direct
python main.py

# Option B: With auto-reload (development)
uvicorn main:app --reload --port 8000
```

The API will be available at `http://localhost:8000`.

### Interactive Docs

FastAPI auto-generates interactive API documentation:
- **Swagger UI:** [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc:** [http://localhost:8000/redoc](http://localhost:8000/redoc)

### Quick Test

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 55, "gender": "M", "restingBP": 120,
    "cholesterol": 198, "fastingBS": 0,
    "chestPainType": "ASY", "restingECG": "Normal",
    "maxHR": 150, "exerciseAngina": "N",
    "oldpeak": 1.5, "stSlope": "Flat"
  }'
```

Expected response:
```json
{"riskLevel": "High Risk", "probability": 80, "rhythm": "Normal Sinus Rhythm"}
```

---

## Deployment (Render)

The backend is deployed on **Render** as a web service.

- **Live URL:** `https://hfp-backend-bqml.onrender.com`
- **Build Command:** `pip install -r requirements.txt`
- **Start Command:** Defined in `Procfile`
- **Environment:** Python 3

The frontend `.env` points to this URL:
```
VITE_API_URL=https://hfp-backend-bqml.onrender.com
```

### Redeployment

Push changes to the connected Git repository. Render will automatically rebuild and redeploy.

```bash
git add .
git commit -m "Update backend"
git push origin main
```

---
