import joblib
import pandas as pd
from flask import Flask, request, jsonify, render_template

# Initialize Flask app
app = Flask(__name__)

# Load preprocessor and XGBoost model
preprocessor = joblib.load('../models/preprocessor.pkl')
model = joblib.load('../models/xgboost_best.pkl')   # or 'final_model.pkl' if renamed

# Expected feature order (must match training)
feature_order = ['crop', 'temperature', 'rainfall', 'sunshine_hours', 'ph',
                 'organic_carbon', 'cec', 'awc', 'bulk_density', 'texture_code']

@app.route('/')
def home():
    return "Crop Suitability Prediction API (XGBoost) is running."

@app.route('/predict', methods=['POST'])
def predict():
    """
    Expects a JSON payload with all required features.
    """
    try:
        data = request.get_json()

        # Validate required features
        missing = [f for f in feature_order if f not in data]
        if missing:
            return jsonify({'error': f'Missing features: {missing}'}), 400

        # Convert to DataFrame with correct column order
        input_df = pd.DataFrame([data])[feature_order]

        # Preprocess
        input_processed = preprocessor.transform(input_df)

        # Predict probability and class
        proba = model.predict_proba(input_processed)[0, 1]   # probability of 'Suitable'
        pred_class = model.predict(input_processed)[0]       # 0 or 1

        # Map to human‑readable label
        suitability = 'Suitable' if pred_class == 1 else 'Unsuitable'

        return jsonify({
            'suitability': suitability,
            'probability': round(proba, 4),
            'input': data
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Optional GET endpoint for quick testing
@app.route('/test_predict', methods=['GET'])
def debug_predict():          # ← renamed to avoid pytest collection
    try:
        data = {
            'crop': request.args.get('crop'),
            'temperature': float(request.args.get('temperature')),
            'rainfall': float(request.args.get('rainfall')),
            'sunshine_hours': float(request.args.get('sunshine_hours')),
            'ph': float(request.args.get('ph')),
            'organic_carbon': float(request.args.get('organic_carbon')),
            'cec': float(request.args.get('cec')),
            'awc': float(request.args.get('awc')),
            'bulk_density': float(request.args.get('bulk_density')),
            'texture_code': int(request.args.get('texture_code'))
        }
        input_df = pd.DataFrame([data])[feature_order]
        input_processed = preprocessor.transform(input_df)
        proba = model.predict_proba(input_processed)[0, 1]
        pred_class = model.predict(input_processed)[0]
        suitability = 'Suitable' if pred_class == 1 else 'Unsuitable'
        return jsonify({
            'suitability': suitability,
            'probability': round(proba, 4),
            'input': data
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':

    app.run(debug=True)