import joblib
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)

TEXTURE_CLASS_MAP = {
    "sand":              1, 
    "loamy sand":        2, 
    "sandy loam":        3, 
    "loam":              4,    
    "silt loam":         5,
    "sandy clay loam":   6,   
    "clay loam":         6,   
    "clay":              7,
}

DEFAULT_TEXTURE_CODE = 4

FEATURE_ORDER = [
    "crop",
    "temperature",
    "rainfall",
    "sunshine_hours",
    "ph",
    "organic_carbon",
    "cec",
    "awc",
    "bulk_density",
    "texture_code",
]

MODELS_DIR    = os.path.join(os.path.dirname(__file__), "models")
PREPROCESSOR  = joblib.load(os.path.join(MODELS_DIR, "preprocessor.pkl"))
MODEL         = joblib.load(os.path.join(MODELS_DIR, "xgboost_best.pkl"))


@app.route("/")
def home():
    return "Crop Suitability Prediction API running on port 5003"


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # -- Resolve texture_code ------------------------------------------
        if "texture_class" in data:
            key = str(data["texture_class"]).strip().lower()
            texture_code = TEXTURE_CLASS_MAP.get(key, DEFAULT_TEXTURE_CODE)
            if TEXTURE_CLASS_MAP.get(key) is None:
                # Log the unknown class but continue with the default
                print(f"[WARN] Unknown texture_class '{data['texture_class']}' - "
                      f"defaulting to texture_code {DEFAULT_TEXTURE_CODE} (loam)")
        elif "texture_code" in data:
            texture_code = int(data["texture_code"])
        else:
            return jsonify({"error": "Either texture_class or texture_code is required"}), 400

        # -- Check required numeric fields ---------------------------------
        numeric_fields = [
            "crop", "temperature", "rainfall", "sunshine_hours",
            "ph", "organic_carbon", "cec", "awc", "bulk_density"
        ]
        missing = [f for f in numeric_fields if f not in data]
        if missing:
            return jsonify({"error": f"Missing fields: {missing}"}), 400

        # -- Build DataFrame -----------------------------------------------
        row = {f: data[f] for f in numeric_fields}
        row["texture_code"] = texture_code

        input_df        = pd.DataFrame([row])[FEATURE_ORDER]
        input_processed = PREPROCESSOR.transform(input_df)

        proba      = MODEL.predict_proba(input_processed)[0, 1]
        pred_class = MODEL.predict(input_processed)[0]
        suitability = "Suitable" if pred_class == 1 else "Unsuitable"

        return jsonify({
            "prediction":           suitability,
            "probability_suitable": round(float(proba), 4),
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5003)
