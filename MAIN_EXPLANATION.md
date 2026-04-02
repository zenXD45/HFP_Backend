# Backend `main.py` Explanation

This file acts as the core server using **FastAPI**. Its main responsibility is to receive patient health data from the frontend, process that data to match how the Machine Learning model was trained, feed the data to the model, and send the prediction results back.

### 1. Setup and Model Loading
```python
app = FastAPI(title="Heart Failure Prediction API")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "rf_heart_failure_model.pkl")

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    model = joblib.load(MODEL_PATH)
```
- **FastAPI Initialization:** Creates the web server application.
- **Model Loading:** It dynamically locates the pre-trained Random Forest model (`rf_heart_failure_model.pkl`) in the same directory using `os.path`. It uses `joblib` to load the model into memory. The `warnings` module is used to suppress potential `scikit-learn` version mismatch warnings, which is a common occurrence when a model is trained in a slightly different environment than where it is deployed.

### 2. CORS Handling (Cross-Origin Resource Sharing)
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    # ...
)
```
Because your frontend (`localhost:5173`) and backend (`localhost:8000`) run on different ports, the browser would normally block requests between them for security. This middleware is configured to allow requests from any origin (`"*"`) so your React frontend can successfully send data to this API.

### 3. Data Validation (Pydantic Model)
```python
class PatientData(BaseModel):
    age: int
    gender: str
    # ...
```
FastAPI uses Pydantic for data validation. `PatientData` defines the exact structure and data types (like `int`, `str`, `float`) of the JSON payload the API expects from the frontend. If the frontend sends incomplete or incorrectly typed data, FastAPI will automatically reject it with an error before it even reaches your prediction logic.

### 4. The API Endpoints
The backend exposes two routes:

#### Health Check (`GET /`)
```python
@app.get("/")
def read_root():
    return {"status": "Heart Failure Prediction API is running with ML model loaded."}
```
A simple endpoint to verify the server is running and the model is successfully loaded.

#### Prediction Engine (`POST /predict`)
```python
@app.post("/predict")
def predict_risk(data: PatientData):
    # ...
```
This is the core logic. When the frontend submits the form, it executes the following steps:

1. **Data Preprocessing & Encoding:** Machine learning models only understand numbers, not strings like "M" or "ASY". The original training data was "One-Hot Encoded". This script manually replicates that encoding for the incoming real-time data:
   - **Ordinal Mapping:** Converts `ST_Slope` terms ("Down", "Flat", "Up") into standard integers (2, 1, 0).
   - **One-Hot Encoding:** It expands categorical variables into binary flags (0 or 1). For example, if `data.gender == 'M'`, then `Sex_M` becomes `1` and `Sex_F` becomes `0`. It does this for Chest Pain Type, Resting ECG, and Exercise Angina.

2. **Feature Alignment:**
   ```python
   feature_order = ['ST_Slope', 'Sex_F', 'Sex_M', ...]
   df = pd.DataFrame([{col: features[col] for col in feature_order}])
   ```
   The Random Forest model requires the input columns to be in the *exact* same order as they were during training. This creates a single-row Pandas DataFrame ensuring perfect alignment.

3. **Inference (Making the Prediction):**
   ```python
   prediction = model.predict(df)[0]
   prob = model.predict_proba(df)[0]
   ```
   - `predict`: Outputs `0` (No Heart Failure) or `1` (Heart Failure).
   - `predict_proba`: Outputs the statistical confidence/probability array (e.g., `[0.2, 0.8]`).

4. **Formatting the Output:**
   - Calculates the risk percentage by looking at `prob[1]` (the probability of the positive class) multiplied by 100.
   - Converts the `0` or `1` into human-readable strings `"Low Risk"` or `"High Risk"`.
   - Generates a descriptive `"rhythm"` string based on the provided ECG reading.
   - Returns these 3 data points as a JSON response to the frontend.

### 5. Running the Application
```python
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```
When you run `python main.py`, this block triggers the `uvicorn` web server to host the application locally on port `8000`.

### Summary
In short, **`main.py` validates incoming HTTP requests, translates human-readable form data into numerical structures the ML model understands, executes the model prediction, and structures the mathematical output back into a user-friendly JSON response.**
