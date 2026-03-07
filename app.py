from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd

app = Flask(__name__)
CORS(app)

pipe = joblib.load("rf_date_season_location.pkl")  # your pipeline model

@app.get("/")
def home():
    return "Backend running "

@app.get("/schema")
def schema():
    return jsonify({
        "inputs": ["date", "season", "location_id"],
        "example": {"date":"2026-02-28","season":"NE-Monsoon","location_id":"1"}
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

    X = pd.DataFrame([{
        "location_id": str(data["location_id"]),
        "season": str(data["season"]),
        "month": int(dt.month),
        "day_of_year": int(dt.dayofyear),
        "year": int(dt.year)
    }])

    pred = float(pipe.predict(X)[0])
    return jsonify({"prediction": round(pred, 2),
                   "rainfall_mm": 1200.00,
                   "sunshine_h":6.5})

if __name__ == "__main__":
    app.run(port=5000, debug=True)