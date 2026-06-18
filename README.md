# Heart Failure Prediction — Backend

A FastAPI-powered REST API that predicts heart failure risk using a trained Random Forest classifier with Recursive Feature Elimination (RFE).

---

## Table of Contents

- [Project Structure](#project-structure)
- [How the Prediction Works](#how-the-prediction-works)
- [The Four Model Files (.pkl)](#the-four-model-files-pkl)
- [API Reference](#api-reference)
- [Code Walkthrough](#code-walkthrough-mainpy)
- [Test Scripts](#test-scripts)
- [Configuration](#configuration)
- [Getting Started](#getting-started)
- [Deployment](#deployment-render)

---

## Project Structure

```
backend/
├── main.py                           → Core API server
├── preprocessor.pkl                  → Data encoder
├── rfe_selector.pkl                  → Feature selector
├── selected_features_rfe_names.pkl   → Feature name list
├── rf_rfe_heart_failure_model.pkl    → Trained model
├── test_api.py                       → HTTP integration test
├── test_function_p.py                → Direct function test
├── requirements.txt                  → Python dependencies
├── Procfile                          → Render deploy command
└── README.md
```

---

## How the Prediction Works

When a patient's data hits the `/predict` endpoint, it passes through three stages:

```
Raw Patient Data (11 fields)
        │
        ▼
┌─────────────────────┐
│  Stage 1: Encode    │  preprocessor.pkl (or manual fallback)
│  11 fields → 18     │  One-hot + ordinal encoding
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Stage 2: Select    │  selected_features_rfe_names.pkl
│  18 → 10 features   │  Keep only what matters
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Stage 3: Predict   │  rf_rfe_heart_failure_model.pkl
│  10 features → Risk │  Random Forest classification
└─────────────────────┘
        │
        ▼
JSON Response: { riskLevel, probability, rhythm }
```

**Why RFE?** Recursive Feature Elimination identified the 10 most predictive features out of 18. This reduces overfitting, improves accuracy, and speeds up inference.

**The 10 features kept:**
`ST_Slope`, `Sex_F`, `ChestPainType_ASY`, `ExerciseAngina_N`, `ExerciseAngina_Y`, `Age`, `RestingBP`, `Cholesterol`, `MaxHR`, `Oldpeak`

**The 8 features dropped:**
`Sex_M`, `ChestPainType_ATA`, `ChestPainType_NAP`, `ChestPainType_TA`, `RestingECG_LVH`, `RestingECG_Normal`, `RestingECG_ST`, `FastingBS`

---

## The Four Model Files (.pkl)

All four files are serialized with `joblib` and loaded once at server startup.

### `preprocessor.pkl` — Data Encoder

- **What:** A `sklearn.compose.ColumnTransformer` (~5 KB)
- **Does:** Converts raw patient strings (`"M"`, `"ASY"`, `"Normal"`) into 18 numeric columns via one-hot and ordinal encoding.
- **Fallback:** If this fails to load (sklearn version mismatch), the server automatically uses `_manual_encode()` — a pure-Python function that produces identical output. The API never breaks.

### `selected_features_rfe_names.pkl` — Feature Name List

- **What:** A plain Python list of 10 strings (~143 bytes)
- **Does:** Tells the server which 10 column names to pick from the encoded DataFrame.
- **Why it exists:** This is the simplest, most portable way to select features — just column names, zero sklearn dependency. It's the **primary** feature selection method.

```python
['ST_Slope', 'Sex_F', 'ChestPainType_ASY', 'ExerciseAngina_N',
 'ExerciseAngina_Y', 'Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
```

### `rfe_selector.pkl` — Feature Selector

- **What:** A fitted `sklearn.feature_selection.RFE` object (~4.5 MB)
- **Does:** Stores the boolean mask (`support_`) and ranking of all 18 features from training.
- **Role:** Acts as a **fallback** if the feature name list can't match columns. Its `.transform()` method selects features by position instead of name.

### `rf_rfe_heart_failure_model.pkl` — The Model

- **What:** A trained `sklearn.ensemble.RandomForestClassifier` (~3.7 MB)
- **Does:** Takes exactly 10 features and outputs a prediction.
  - `predict()` → `0` (Low Risk) or `1` (High Risk)
  - `predict_proba()` → e.g. `[0.20, 0.80]` — the value at index 1 becomes the percentage shown to users.

---

## API Reference

### `GET /` — Health Check

Returns the server status and list of features the model uses.

```json
{
  "status": "Heart Failure Prediction API is running with ML model loaded.",
  "model": "Random Forest (RFE-selected features)",
  "features_used": ["ST_Slope", "Sex_F", "ChestPainType_ASY", ...]
}
```

### `POST /predict` — Predict Risk

**Request:**

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

| Field | Type | Values |
|-------|------|--------|
| `age` | int | Patient age |
| `gender` | str | `"M"` / `"F"` |
| `restingBP` | int | Resting blood pressure (mm Hg) |
| `cholesterol` | int | Serum cholesterol (mg/dl) |
| `fastingBS` | int | `1` if fasting blood sugar > 120, else `0` |
| `chestPainType` | str | `"ASY"` / `"ATA"` / `"NAP"` / `"TA"` |
| `restingECG` | str | `"Normal"` / `"LVH"` / `"ST"` |
| `maxHR` | int | Max heart rate achieved |
| `exerciseAngina` | str | `"Y"` / `"N"` |
| `oldpeak` | float | ST depression value |
| `stSlope` | str | `"Up"` / `"Flat"` / `"Down"` |

**Response:**

```json
{
  "riskLevel": "High Risk",
  "probability": 80,
  "rhythm": "Normal Sinus Rhythm"
}
```

| Field | Description |
|-------|-------------|
| `riskLevel` | `"High Risk"` or `"Low Risk"` |
| `probability` | Heart failure probability (0–100%) |
| `rhythm` | ECG rhythm: `"Normal Sinus Rhythm"`, `"Left Ventricular Hypertrophy"`, or `"ST-T Wave Abnormality"` |

---

## Code Walkthrough — `main.py`

The entire backend is a single file with clear sections:

**Imports & Setup** — FastAPI app, joblib for model loading, pandas for data handling, warnings to suppress sklearn noise.

**Artifact Loading (runs once at startup):**
- Loads all 4 `.pkl` files into memory
- Wraps `preprocessor.pkl` in try/except for version safety
- If it fails, sets `preprocessor = None` to trigger the manual fallback path

**CORS Middleware** — Allows cross-origin requests from the frontend (`localhost:5173` in dev, or Capacitor mobile apps). Without this, browsers block the requests.

**`PatientData` (Pydantic model)** — Defines the 11 input fields with types. FastAPI auto-validates incoming requests against this schema and returns `422` if anything is wrong.

**`_manual_encode(data)`** — Pure Python fallback encoder. Replicates the ColumnTransformer's logic:
- Ordinal-encodes ST_Slope: Up → 0, Flat → 1, Down → 2
- One-hot encodes gender, chest pain type, resting ECG, exercise angina
- Passes through numeric fields unchanged
- Returns an 18-column DataFrame

**`_preprocess(data)`** — Tries `preprocessor.pkl` first, falls back to `_manual_encode()`.

**`_select_rfe_features(df)`** — Picks the 10 RFE features. Tries column name matching first (using the name list), falls back to the RFE selector's `.transform()`.

**`predict_risk(data)` (`POST /predict`)** — The main endpoint:
1. Encode → 18 features
2. Select → 10 features
3. Predict → class + probability
4. Map rhythm from `restingECG` input
5. Return JSON

**Entry point** — `python main.py` starts Uvicorn on port 8000.

---

## Test Scripts

### `test_function_p.py`

Tests the prediction function directly (no HTTP). Import `predict_risk`, pass a `PatientData` object, print the result.

```bash
python test_function_p.py
```

### `test_api.py`

Tests the API over HTTP with a sample patient payload. Requires the server to be running.

```bash
python main.py &        # Start server
python test_api.py      # Run test
```

---

## Configuration

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

`scikit-learn` is pinned to `1.6.1` to match the training environment and avoid `.pkl` deserialization issues.

### `Procfile`

```
web: uvicorn main:app --host 0.0.0.0 --port ${PORT:-10000}
```

Used by Render to start the app. Reads the `PORT` env variable, defaults to 10000.

---

## Getting Started

```bash
# 1. Go to the backend folder
cd backend

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start the server
python main.py
```

The API is now live at **http://localhost:8000**.

Auto-generated docs are available at:
- Swagger UI → http://localhost:8000/docs
- ReDoc → http://localhost:8000/redoc

**Quick test:**

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age":55,"gender":"M","restingBP":120,"cholesterol":198,"fastingBS":0,"chestPainType":"ASY","restingECG":"Normal","maxHR":150,"exerciseAngina":"N","oldpeak":1.5,"stSlope":"Flat"}'
```

---

## Deployment (Render)

The backend is live on Render:

- **URL:** `https://hfp-backend-bqml.onrender.com`
- **Build:** `pip install -r requirements.txt`
- **Start:** Defined in `Procfile`


To redeploy, just push to the connected Git repo:

```bash
git add .
git commit -m "Update backend"
git push origin main
```

Render rebuilds automatically.
