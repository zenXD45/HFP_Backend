import requests

data = {
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

try:
    # First ensure the server is running in background
    pass
except Exception as e:
    pass
