from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
import os

app = Flask(__name__)
CORS(app)

# =========================================================
# MODEL PATHS
# =========================================================
_DIR = os.path.dirname(__file__)

TEMP_MODEL_PATH = os.path.join(_DIR, "rf_seasonal_temperature.pkl")
RAIN_MODEL_PATH = os.path.join(_DIR, "rf_seasonal_rainfall.pkl")
SUN_MODEL_PATH  = os.path.join(_DIR, "rf_seasonal_sunshine.pkl")

for path in [TEMP_MODEL_PATH, RAIN_MODEL_PATH, SUN_MODEL_PATH]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")

temp_model = joblib.load(TEMP_MODEL_PATH)
rain_model = joblib.load(RAIN_MODEL_PATH)
sun_model  = joblib.load(SUN_MODEL_PATH)


# =========================================================
# HOME
# =========================================================
@app.get("/")
def home():
    return "Weather Prediction API running on port 5002"


# =========================================================
# SCHEMA
# =========================================================
@app.get("/schema")
def schema():
    return jsonify({
        "features": ["season", "year", "location_id"],
        "example": {
            "season":      "North-east monsoon",
            "year":        2026,
            "location_id": "1"
        },
        "valid_seasons": [
            "South-west monsoon",
            "Intermonsoon after South-west monsoon",
            "North-east monsoon",
            "Intermonsoon after North-east monsoon"
        ],
        "location_id_map": {
            "1": "Colombo  (lat 6.72 - 7.00)",
            "2": "Gampaha  (lat 7.00 - 7.30)",
            "3": "Kalutara (lat 6.40 - 6.72)"
        }
    })


# =========================================================
# PREDICT
# =========================================================
@app.post("/predict")
def predict():
    data = request.get_json(silent=True) or {}

    location_id = data.get("location_id")
    if not location_id:
        return jsonify({"error": "Missing required field: location_id"}), 400

    season = data.get("season")
    year   = data.get("year")

    if season is None or year is None:
        return jsonify({
            "error": "Provide either (season + year + location_id) or (date + location_id)"
        }), 400

    try:
        year = int(year)
    except Exception:
        return jsonify({"error": "Year must be an integer"}), 400

    season      = str(season).strip()
    location_id = str(location_id).strip()

    valid_seasons = [
        "South-west monsoon",
        "Intermonsoon after South-west monsoon",
        "North-east monsoon",
        "Intermonsoon after North-east monsoon",
    ]
    if season not in valid_seasons:
        return jsonify({
            "error":         "Invalid season value",
            "valid_seasons": valid_seasons
        }), 400

    X = pd.DataFrame([{
        "location_id": location_id,
        "season":      season,
        "year":        year,
    }])

    pred_temp     = float(temp_model.predict(X)[0])
    pred_rain_log = float(rain_model.predict(X)[0])
    pred_rain     = float(np.expm1(pred_rain_log))  
    pred_sun      = float(sun_model.predict(X)[0])

    return jsonify({
        "season":        season,
        "year":          year,
        "location_id":   location_id,
        "temperature_C": round(pred_temp, 2),
        "rainfall_mm":   round(pred_rain, 2),
        "sunshine_h":    round(pred_sun,  2),
    })


if __name__ == "__main__":
    app.run(port=5002, debug=True)
