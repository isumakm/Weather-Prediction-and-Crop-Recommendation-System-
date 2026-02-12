from pathlib import Path
import pandas as pd
from pathlib import Path
from .db import get_connection

csv_path = Path(r"C:\Users\chama\Weather-Prediction-and-Crop-Recommendation-System-\soil_suitability\outputs\soil_features_for_db_final.csv")

df = pd.read_csv(csv_path)

conn = get_connection()
cursor = conn.cursor()

query = """
INSERT INTO soil_points
(lat, lon, taw, organic_carbon, cec, ph, sand_pct, bulk_density, awc, texture_class, cluster)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
"""

for _, row in df.iterrows():
    cursor.execute(query, (
        row["lat"],
        row["lon"],
        row["taw"],
        row["organic_carbon"],
        row["cec"],
        row["ph"],
        row["sand_pct"],
        row["bulk_density"],     
        row["awc"],              
        row["texture_class"],   
        row["cluster"]
    ))

conn.commit()
cursor.close()
conn.close()

print("Soil data inserted successfully.")
