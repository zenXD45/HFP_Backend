<![CDATA[<div align="center">

# 🫀 Heart Failure Prediction — Backend

**An intelligent REST API that predicts heart failure risk using machine learning**

[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org/)
[![Render](https://img.shields.io/badge/Deployed_on-Render-46E3B7?style=for-the-badge&logo=render&logoColor=white)](https://render.com/)

*Receives patient clinical data → Processes through a trained Random Forest pipeline with Recursive Feature Elimination → Returns a heart-failure risk prediction in milliseconds.*

---

[🚀 Quick Start](#-local-development) · [📡 API Docs](#-api-reference) · [🧠 How It Works](#-ml-pipeline-deep-dive) · [📦 Model Files](#-model-artifacts-the-four-pkl-files)

</div>

---

## 📂 Project Structure

```
backend/
│
├── 🐍 main.py                           # Core FastAPI server — all routes & ML logic
│
├── 🧪 Model Artifacts (The Brain)
│   ├── preprocessor.pkl                  # Sklearn ColumnTransformer (~5 KB)
│   ├── rfe_selector.pkl                  # Sklearn RFE selector (~4.5 MB)
│   ├── selected_features_rfe_names.pkl   # List of 10 feature names (~143 B)
│   └── rf_rfe_heart_failure_model.pkl    # Trained Random Forest (~3.7 MB)
│
├── 🧰 Test Suite
│   ├── test_api.py                       # HTTP-level integration test
│   └── test_function_p.py               # Direct function-level unit test
│
├── ⚙️  Config
│   ├── requirements.txt                  # Pinned Python dependencies
│   └── Procfile                          # Render deployment entrypoint
│
└── 📖 README.md                          # You are here!
```

---

## 🧠 ML Pipeline Deep Dive

Every prediction request flows through a **3-stage pipeline** — each stage powered by a dedicated `.pkl` artifact:

```
  📥 Raw Patient Data (11 fields from the frontend)
         │
         ▼
  ┌─────────────────────────────────────────┐
  │  STAGE 1 · PREPROCESS                  │
  │  ─────────────────────────              │
  │  📄 preprocessor.pkl                    │
  │                                         │
  │  • One-hot encodes categorical fields   │
  │  • Ordinal-maps ST_Slope               │
  │  • Passes through numeric fields        │
  │                                         │
  │  Input:  11 raw fields                  │
  │  Output: 18 encoded features            │
  └──────────────────┬──────────────────────┘
                     │
                     ▼
  ┌─────────────────────────────────────────┐
  │  STAGE 2 · FEATURE SELECTION            │
  │  ─────────────────────────              │
  │  📄 selected_features_rfe_names.pkl     │
  │  📄 rfe_selector.pkl (fallback)         │
  │                                         │
  │  • Picks the 10 most predictive         │
  │    features identified by RFE           │
  │  • Drops 8 noisy/redundant features     │
  │                                         │
  │  Input:  18 encoded features            │
  │  Output: 10 selected features           │
  └──────────────────┬──────────────────────┘
                     │
                     ▼
  ┌─────────────────────────────────────────┐
  │  STAGE 3 · PREDICTION                  │
  │  ─────────────────────────              │
  │  📄 rf_rfe_heart_failure_model.pkl      │
  │                                         │
  │  • Random Forest classifier             │
  │  • Outputs binary class + probability   │
  │                                         │
  │  Input:  10 selected features           │
  │  Output: Risk level + % probability     │
  └─────────────────────────────────────────┘
         │
         ▼
  📤 JSON Response → { riskLevel, probability, rhythm }
```

### 🎯 Why Recursive Feature Elimination (RFE)?

RFE was applied during model training to surgically identify the **10 most informative features** from 18 total. The benefits:

| Benefit | Impact |
|---------|--------|
| 🎯 **Higher accuracy** | Removes noisy features that confuse the model |
| ⚡ **Faster inference** | 44% fewer features = faster predictions |
| 🛡️ **Reduced overfitting** | Fewer parameters = better generalization |

### ✅ The 10 Selected Features (Kept by RFE)

| # | Feature | Type | What it Represents |
|---|---------|------|--------------------|
| 1 | `ST_Slope` | Ordinal | ST segment slope (Up=0, Flat=1, Down=2) |
| 2 | `Sex_F` | Binary | Patient is female |
| 3 | `ChestPainType_ASY` | Binary | Asymptomatic chest pain |
| 4 | `ExerciseAngina_N` | Binary | No exercise-induced angina |
| 5 | `ExerciseAngina_Y` | Binary | Has exercise-induced angina |
| 6 | `Age` | Numeric | Patient age in years |
| 7 | `RestingBP` | Numeric | Resting blood pressure (mm Hg) |
| 8 | `Cholesterol` | Numeric | Serum cholesterol (mg/dl) |
| 9 | `MaxHR` | Numeric | Maximum heart rate achieved |
| 10 | `Oldpeak` | Numeric | ST depression (exercise vs rest) |

### ❌ The 8 Eliminated Features (Dropped by RFE)

> `Sex_M` · `ChestPainType_ATA` · `ChestPainType_NAP` · `ChestPainType_TA` · `RestingECG_LVH` · `RestingECG_Normal` · `RestingECG_ST` · `FastingBS`

These were ranked lower by RFE and removed to improve model performance.

---

## 📦 Model Artifacts — The Four `.pkl` Files

All `.pkl` files are serialized Python objects saved with `joblib`. They are loaded **once** at server startup and held in memory for instant predictions.

---

### 1️⃣ `preprocessor.pkl` — The Encoder

| Property | Value |
|----------|-------|
| **Type** | `sklearn.compose.ColumnTransformer` |
| **Size** | ~5 KB |
| **Loaded at** | Server startup (with `try/except` fallback) |

**What it does:** Transforms raw patient data (with string categorical fields like `"M"`, `"ASY"`, `"Normal"`) into a fully numeric 18-column DataFrame via one-hot encoding and ordinal mapping.

**Resilience:** This artifact may fail to load if the deployment environment has a different `scikit-learn` version than the training environment. The backend handles this gracefully — if loading fails, it sets `preprocessor = None` and uses the `_manual_encode()` function, which produces **identical output** using pure Python logic.

```python
# Fallback: No sklearn needed
'Sex_F': 1 if data.gender == 'F' else 0,
'Sex_M': 1 if data.gender == 'M' else 0,
# ... and so on for all 18 features
```

---

### 2️⃣ `rfe_selector.pkl` — The Feature Filter

| Property | Value |
|----------|-------|
| **Type** | `sklearn.feature_selection.RFE` |
| **Size** | ~4.5 MB |
| **Role** | Secondary fallback for feature selection |

**What it does:** A fitted Recursive Feature Elimination object that knows which 10 of the 18 features to keep. Internally it stores:

- `support_` — Boolean mask of selected features
- `ranking_` — Importance ranking of all 18 features
- `feature_names_in_` — The 18 input feature names it expects

**When it's used:** This object acts as a **fallback**. The primary feature selection method uses `selected_features_rfe_names.pkl` (a simple list of column names). If that fails (e.g., column name mismatch), the RFE selector's `.transform()` method handles positional selection.

---

### 3️⃣ `selected_features_rfe_names.pkl` — The Feature Checklist

| Property | Value |
|----------|-------|
| **Type** | Python `list` of 10 strings |
| **Size** | ~143 bytes |
| **Role** | Primary feature selection method |

**What it does:** The simplest and most portable artifact — just a Python list:

```python
['ST_Slope', 'Sex_F', 'ChestPainType_ASY', 'ExerciseAngina_N',
 'ExerciseAngina_Y', 'Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
```

**Why it exists:** Selecting features by column name is the safest approach — it has **zero sklearn version dependency** and works as long as the column names match. This is why it's the primary method over the RFE selector.

---

### 4️⃣ `rf_rfe_heart_failure_model.pkl` — The Predictor

| Property | Value |
|----------|-------|
| **Type** | `sklearn.ensemble.RandomForestClassifier` |
| **Size** | ~3.7 MB |
| **Role** | Core prediction engine |

**What it does:** The trained Random Forest classification model. It expects exactly **10 features** (the RFE-selected ones) in the correct order and outputs:

| Method | Output | Example |
|--------|--------|---------|
| `model.predict()` | `0` (No Heart Failure) or `1` (Heart Failure) | `1` |
| `model.predict_proba()` | Probability array for each class | `[0.20, 0.80]` |

The probability at index `[1]` (positive class) is multiplied by 100 and returned as the `probability` percentage in the API response.

---

### 🔄 Artifact Loading Strategy — Resilience by Design

```
Server Startup
  │
  ├─ Load selected_features_rfe_names.pkl  ✅ Always works (plain list)
  │
  ├─ Load preprocessor.pkl
  │   ├─ Success → Use ColumnTransformer
  │   └─ Failure → Set to None, use _manual_encode() fallback
  │
  ├─ Load rfe_selector.pkl                 ✅ Fallback for feature selection
  │
  └─ Load rf_rfe_heart_failure_model.pkl   ✅ Core model
```

This layered approach means the API **never crashes** due to sklearn version mismatches. It degrades gracefully through multiple fallback paths.

---

## 📡 API Reference

### `GET /` — Health Check

Verifies the server is running and the model is loaded.

```bash
curl https://hfp-backend-bqml.onrender.com/
```

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

**Input Fields:**

| Field | Type | Accepted Values |
|-------|------|-----------------|
| `age` | `int` | Patient age (e.g., `55`) |
| `gender` | `str` | `"M"` or `"F"` |
| `restingBP` | `int` | Resting blood pressure in mm Hg |
| `cholesterol` | `int` | Serum cholesterol in mg/dl |
| `fastingBS` | `int` | Fasting blood sugar > 120 mg/dl: `1` = true, `0` = false |
| `chestPainType` | `str` | `"ASY"`, `"ATA"`, `"NAP"`, or `"TA"` |
| `restingECG` | `str` | `"Normal"`, `"LVH"`, or `"ST"` |
| `maxHR` | `int` | Maximum heart rate achieved |
| `exerciseAngina` | `str` | `"Y"` or `"N"` |
| `oldpeak` | `float` | ST depression value (e.g., `1.5`) |
| `stSlope` | `str` | `"Up"`, `"Flat"`, or `"Down"` |

**Response:**
```json
{
  "riskLevel": "High Risk",
  "probability": 80,
  "rhythm": "Normal Sinus Rhythm"
}
```

| Response Field | Type | Description |
|---------------|------|-------------|
| `riskLevel` | `str` | `"High Risk"` (prediction=1) or `"Low Risk"` (prediction=0) |
| `probability` | `int` | Heart failure probability as a percentage (0–100) |
| `rhythm` | `str` | ECG rhythm description based on `restingECG` input |

**Rhythm Mapping:**

| `restingECG` Input | Rhythm Output |
|--------------------|---------------|
| `"Normal"` | `"Normal Sinus Rhythm"` |
| `"LVH"` | `"Left Ventricular Hypertrophy"` |
| `"ST"` | `"ST-T Wave Abnormality"` |

---

## 🐍 Code Walkthrough — `main.py`

The entire backend lives in a single, well-organized file. Here's how each section works:

### 1. Imports & App Initialization

```python
import warnings, joblib, numpy as np, pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os

app = FastAPI(title="Heart Failure Prediction API")
```

| Library | Purpose |
|---------|---------|
| `FastAPI` | Creates the web server and handles routing |
| `joblib` | Loads serialized `.pkl` model artifacts |
| `pandas` | Constructs DataFrames matching the model's expected input |
| `warnings` | Suppresses sklearn version mismatch warnings |
| `pydantic` | Automatic request validation via `BaseModel` |

### 2. Artifact Loading (Startup)

```python
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    selected_features = joblib.load(...)  # Feature name list
    try:
        preprocessor = joblib.load(...)   # ColumnTransformer (may fail)
    except Exception:
        preprocessor = None               # Graceful fallback
    rfe_selector = joblib.load(...)       # RFE selector
    model = joblib.load(...)              # Random Forest model
```

All 4 `.pkl` files are loaded **once** at startup into memory — no disk I/O on each request. The `preprocessor.pkl` load is wrapped in `try/except` because `ColumnTransformer` objects are sensitive to sklearn version changes.

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

The frontend (e.g., `localhost:5173` or a Capacitor mobile app) and backend run on different origins. Without CORS middleware, browsers block cross-origin requests. This configuration allows requests from **any** origin.

### 4. Request Schema (Pydantic)

```python
class PatientData(BaseModel):
    age: int
    gender: str
    restingBP: int
    # ... 11 fields total
```

FastAPI uses Pydantic models for **automatic request validation**. If the frontend sends missing or incorrectly typed fields, FastAPI returns a `422 Unprocessable Entity` error *before* any prediction logic runs.

### 5. `_manual_encode()` — Fallback Encoder

Replicates the `preprocessor.pkl` ColumnTransformer using pure Python — no sklearn dependency:

- **Ordinal Encoding** for `ST_Slope`: `"Up"` → `0`, `"Flat"` → `1`, `"Down"` → `2`
- **One-Hot Encoding** for categoricals: `gender="M"` → `Sex_M=1, Sex_F=0`
- **Passthrough** for numerics: `Age`, `RestingBP`, `Cholesterol`, `MaxHR`, `Oldpeak`, `FastingBS`

Returns a single-row DataFrame with **18 columns** in the exact order the pipeline expects.

### 6. `_preprocess()` — Smart Encoder

A wrapper that tries the saved `preprocessor.pkl` first, then falls back to `_manual_encode()` if the preprocessor couldn't be loaded. This dual strategy ensures the backend works regardless of the sklearn version.

### 7. `_select_rfe_features()` — Feature Selector

Takes the full 18-column DataFrame and narrows it to **10 columns**:

1. **Primary:** Direct column selection using `selected_features_rfe_names.pkl` (most portable)
2. **Fallback:** RFE selector's `.transform()` method (position-based)

### 8. `predict_risk()` — The Prediction Endpoint

Orchestrates the full pipeline in 3 steps:

```python
df_encoded  = _preprocess(data)                # Step 1: Encode (18 features)
df_selected = _select_rfe_features(df_encoded)  # Step 2: Select (10 features)
prediction  = model.predict(df_selected)[0]     # Step 3: Predict
prob        = model.predict_proba(df_selected)[0]
```

Then formats the output into `{ riskLevel, probability, rhythm }`.

### 9. Server Entry Point

```python
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

When running `python main.py` directly, Uvicorn starts the ASGI server on port `8000`.

---

## 🧪 Test Scripts

### `test_function_p.py` — Direct Function Test

```python
from main import app, PatientData, predict_risk

data = PatientData(**{"age": 55, "gender": "M", ...})
print(predict_risk(data))
```

Tests the `predict_risk` function **directly** without HTTP. Useful for debugging the ML pipeline in isolation.

```bash
python test_function_p.py
```

### `test_api.py` — HTTP Integration Test

Contains a sample patient payload for testing the API over HTTP. Requires the server to be running first.

```bash
# Terminal 1: Start server
python main.py

# Terminal 2: Run test
python test_api.py
```

---

## ⚙️ Configuration Files

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

> **Note:** `scikit-learn` is pinned to `1.6.1` to match the version used during model training — this minimizes deserialization issues with the `.pkl` files.

### `Procfile`

```
web: uvicorn main:app --host 0.0.0.0 --port ${PORT:-10000}
```

Used by **Render** (and other PaaS platforms like Heroku) to start the application. It uses the platform's `PORT` environment variable, defaulting to `10000`.

---

## 🚀 Local Development

### Prerequisites

- Python 3.10+
- pip

### Setup

```bash
# Navigate to the backend directory
cd backend

# Create & activate virtual environment
python -m venv venv
source venv/bin/activate        # Linux/macOS
# venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt
```

### Run the Server

```bash
# Option A: Direct
python main.py

# Option B: With auto-reload (recommended for development)
uvicorn main:app --reload --port 8000
```

The API will be available at `http://localhost:8000`.

### 📚 Interactive Docs (Auto-Generated)

FastAPI auto-generates beautiful interactive API documentation:

| Docs | URL |
|------|-----|
| **Swagger UI** | [http://localhost:8000/docs](http://localhost:8000/docs) |
| **ReDoc** | [http://localhost:8000/redoc](http://localhost:8000/redoc) |

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

**Expected Response:**
```json
{"riskLevel": "High Risk", "probability": 80, "rhythm": "Normal Sinus Rhythm"}
```

---

## ☁️ Deployment (Render)

The backend is deployed on **Render** as a web service.

| Config | Value |
|--------|-------|
| **Live URL** | `https://hfp-backend-bqml.onrender.com` |
| **Build Command** | `pip install -r requirements.txt` |
| **Start Command** | Defined in `Procfile` |
| **Environment** | Python 3 |

The frontend `.env` points to this URL:
```
VITE_API_URL=https://hfp-backend-bqml.onrender.com
```

### Redeployment

Push changes to the connected Git repository — Render will automatically rebuild and redeploy:

```bash
git add .
git commit -m "Update backend"
git push origin main
```

---

<div align="center">

**Built with ❤️ using FastAPI + scikit-learn**

</div>
]]>
