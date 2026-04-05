from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import os

app = Flask(__name__)
CORS(app)

# =========================================================
# MODEL PATHS
# =========================================================
TEMP_MODEL_PATH = "rf_seasonal_temperature.pkl"
RAIN_MODEL_PATH = "rf_seasonal_rainfall.pkl"
SUN_MODEL_PATH  = "rf_seasonal_sunshine.pkl"

for path in [TEMP_MODEL_PATH, RAIN_MODEL_PATH, SUN_MODEL_PATH]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")

temp_model = joblib.load(TEMP_MODEL_PATH)
rain_model = joblib.load(RAIN_MODEL_PATH)
sun_model = joblib.load(SUN_MODEL_PATH)


# =========================================================
# HOME
# =========================================================
@app.get("/")
def home():
    return "Seasonal weather backend running"


# =========================================================
# SCHEMA
# =========================================================
@app.get("/schema")
def schema():
    return jsonify({
        "features": ["season", "year", "location_id"],
        "example": {
            "season": "North-east monsoon",
            "year": 2026,
            "location_id": "1"
        }
    })


# =========================================================
# OPTIONAL DATE -> SEASON CONVERTER
# =========================================================
def get_sri_lanka_season(month):
    if month in [5, 6, 7, 8, 9]:
        return "South-west monsoon"
    elif month in [10, 11]:
        return "Intermonsoon after South-west monsoon"
    elif month in [12, 1, 2]:
        return "North-east monsoon"
    elif month in [3, 4]:
        return "Intermonsoon after North-east monsoon"
    return "Unknown"


# =========================================================
# PREDICT
# =========================================================
@app.post("/predict")
def predict():
    data = request.get_json(silent=True) or {}

    # Accept either direct seasonal input OR date input
    # Preferred input:
    # {
    #   "season": "North-east monsoon",
    #   "year": 2026,
    #   "location_id": "1"
    # }
    #
    # Optional alternative:
    # {
    #   "date": "2026-02-28",
    #   "location_id": "1"
    # }
    # date will be converted to season automatically

    location_id = data.get("location_id")

    if not location_id:
        return jsonify({
            "error": "Missing required field: location_id"
        }), 400

    season = data.get("season")
    year = data.get("year")

    # If season/year not given, try deriving from date
    if (season is None or year is None) and "date" in data:
        dt = pd.to_datetime(data["date"], errors="coerce")
        if pd.isna(dt):
            return jsonify({
                "error": "Invalid date format. Use YYYY-MM-DD"
            }), 400

        season = get_sri_lanka_season(dt.month)
        year = int(dt.year)

    # Validate final values
    if season is None or year is None:
        return jsonify({
            "error": "Missing required fields",
            "expected_any_one_of": [
                {
                    "season": "North-east monsoon",
                    "year": 2026,
                    "location_id": "1"
                },
                {
                    "date": "2026-02-28",
                    "location_id": "1"
                }
            ]
        }), 400

    try:
        year = int(year)
    except Exception:
        return jsonify({
            "error": "Year must be an integer"
        }), 400

    season = str(season).strip()
    location_id = str(location_id).strip()

    valid_seasons = [
        "South-west monsoon",
        "Intermonsoon after South-west monsoon",
        "North-east monsoon",
        "Intermonsoon after North-east monsoon"
    ]

    if season not in valid_seasons:
        return jsonify({
            "error": "Invalid season value",
            "valid_seasons": valid_seasons
        }), 400

    # Must match training input columns exactly
    X = pd.DataFrame([{
        "location_id": location_id,
        "season": season,
        "year": year
    }])

    # Predict
    pred_temp = float(temp_model.predict(X)[0])

    pred_rain_log = float(rain_model.predict(X)[0])
    pred_rain = float(np.expm1(pred_rain_log))   # convert back from log scale

    pred_sun = float(sun_model.predict(X)[0])

    return jsonify({
        "season": season,
        "year": year,
        "location_id": location_id,
        "temperature_C": round(pred_temp, 2),
        "rainfall_mm": round(pred_rain, 2),
        "sunshine_h": round(pred_sun, 2)
    })


if __name__ == "__main__":
    app.run(port=5000, debug=True)