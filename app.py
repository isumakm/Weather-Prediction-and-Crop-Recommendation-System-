from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import os

app = Flask(__name__)
CORS(app)

MODEL_PATH = "rf_weather_3targets.pkl"   # multi-output model (temp, rain, sunshine)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

pipe = joblib.load(MODEL_PATH)

@app.get("/")
def home():
    return "Backend running"

@app.get("/schema")
def schema():
    # Use 'features' key to match your React schema loader
    return jsonify({
        "features": ["date", "season", "location_id"],
        "example": {"date": "2026-02-28", "season": "NE-Monsoon", "location_id": "1"}
    })

@app.post("/predict")
def predict():
    data = request.get_json(silent=True) or {}

    required = ["date", "season", "location_id"]
    missing = [k for k in required if k not in data]
    if missing:
        return jsonify({
            "error": "Missing required fields",
            "missing": missing,
            "expected_format": {"date": "YYYY-MM-DD", "season": "NE-Monsoon", "location_id": "1"}
        }), 400

    dt = pd.to_datetime(data["date"], errors="coerce")
    if pd.isna(dt):
        return jsonify({"error": "Invalid date format. Use YYYY-MM-DD"}), 400

    # Must match training input columns
    X = pd.DataFrame([{
        "location_id": str(data["location_id"]),
        "season": str(data["season"]),
        "month": int(dt.month),
        "day_of_year": int(dt.dayofyear),
        "year": int(dt.year)
    }])

    # Multi-output prediction
    pred = pipe.predict(X)[0]   # [temp, rain, sunshine]
    temp = float(pred[0])
    rain = float(pred[1])
    sun  = float(pred[2])

    return jsonify({
        "temperature_C": round(temp, 2),
        "rainfall_mm": round(rain, 2),
        "sunshine_h": round(sun, 2)
    })

if __name__ == "__main__":
    app.run(port=5000, debug=True)