from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import os

app = Flask(__name__)
CORS(app)

MODEL = joblib.load(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Notebooks", "tuned_svm.pkl")))

CROP_LIST = [
    'Banana', 'Bitter Gourd', 'Brinjal', 'Capsicum', 'Cucumber', 'Ginger',
    'Kiriala', 'Luffa', 'Mangosteen', 'Manioc', 'Okra', 'Papaya', 'Passion Fruit',
    'Pineapple', 'Radish', 'Rambutan', 'Snake Gourd', 'Sweet Potato', 'Turmeric',
    'Yams', 'Yard Long Bean'
]


REQUIRED = [
    "temperature", "rainfall", "sunshine_hours",
    "ph", "organic_carbon", "cec", "awc", "bulk_density", "texture_code"
]


@app.route("/rank")
def rank():
    missing = [f for f in REQUIRED if request.args.get(f) is None]
    if missing:
        return f"""
        <h2 style="color:red">Missing parameters: {missing}</h2>
        <h3>Example URL - copy into browser and change the values:</h3>
        <a href="/rank?temperature=28.5&rainfall=1200&sunshine_hours=7&ph=6.2&organic_carbon=1.8&cec=12&awc=0.025&bulk_density=1.3&texture_code=3">
        http://localhost:5010/rank?temperature=28.5&rainfall=1200&sunshine_hours=7&ph=6.2&organic_carbon=1.8&cec=12&awc=0.025&bulk_density=1.3&texture_code=3
        </a>
        <br><br>
        <b>texture_code:</b> 1=sand, 2=loamy sand, 3=sandy loam, 4=loam, 5=silt loam, 6=clay loam, 7=clay
        """, 400

    # Parse all values as floats
    try:
        vals = {f: float(request.args.get(f)) for f in REQUIRED}
    except ValueError as e:
        return f"<h2 style='color:red'>Invalid value: {e}</h2>", 400

    # Build 21 rows - one per crop
    rows = [{"crop": c, "scenario_id": 1, **vals} for c in CROP_LIST]
    df    = pd.DataFrame(rows)
    proba = MODEL.predict_proba(df)[:, 1]
    pred  = MODEL.predict(df)

    results = sorted(
        [{"crop": CROP_LIST[i], "probability": round(float(proba[i]) * 100, 1), "suitable": bool(pred[i] == 1)}
         for i in range(len(CROP_LIST))],
        key=lambda x: x["probability"], reverse=True
    )
    for i, r in enumerate(results):
        r["rank"] = i + 1

    # Render a clean HTML table in the browser
    rows_html = ""
    for r in results:
        color = "#2d6a04" if r["suitable"] else "#8b0000"
        label = "Suitable" if r["suitable"] else "Unsuitable"
        rows_html += f"""
        <tr>
            <td style="padding:8px 16px;font-weight:bold">{r['rank']}</td>
            <td style="padding:8px 16px">{r['crop']}</td>
            <td style="padding:8px 16px;font-weight:bold">{r['probability']}%</td>
            <td style="padding:8px 16px;color:{color};font-weight:bold">{label}</td>
        </tr>"""

    inputs_html = "&nbsp;&nbsp;".join([f"<b>{f}</b>={vals[f]}" for f in REQUIRED])

    return f"""<!DOCTYPE html>
<html>
<head>
    <title>Crop Ranking Results</title>
    <style>
        body  {{ font-family: Arial, sans-serif; padding: 30px; background: #f9f7f3; }}
        h2    {{ color: #2d5205; }}
        table {{ border-collapse: collapse; background: white; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
        th    {{ background: #3e6e06; color: white; padding: 10px 16px; text-align: left; }}
        tr:nth-child(even) {{ background: #f0f5e8; }}
        .inputs {{ background: #e8f0d8; padding: 12px 16px; border-radius: 6px; margin-bottom: 20px; font-size: 14px; }}
        .tip    {{ margin-top: 20px; font-size: 13px; color: #555; }}
    </style>
</head>
<body>
    <h2>Multi-Crop Ranking Results</h2>
    <div class="inputs"><b>Inputs used:</b><br><br>{inputs_html}</div>
    <table>
        <tr><th>Rank</th><th>Crop</th><th>Probability</th><th>Prediction</th></tr>
        {rows_html}
    </table>
    <p class="tip">
        To try different values, edit the numbers directly in the browser address bar and press Enter.<br><br>
        <b>texture_code values:</b> 1=sand &nbsp; 2=loamy sand &nbsp; 3=sandy loam &nbsp; 4=loam &nbsp; 5=silt loam &nbsp; 6=clay loam &nbsp; 7=clay
    </p>
</body>
</html>"""


if __name__ == "__main__":
    print("\nMulti-Crop Ranking API is running.")
    print("\nOpen this URL in your browser, then change any values and press Enter:")
    print("http://localhost:5010/rank?temperature=28.5&rainfall=1200&sunshine_hours=7&ph=6.2&organic_carbon=1.8&cec=12&awc=0.025&bulk_density=1.3&texture_code=3\n")
    app.run(debug=True, host="0.0.0.0", port=5010)
