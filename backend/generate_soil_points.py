import pandas as pd
from pathlib import Path

CSV_PATH = Path(
    r"C:\Users\chama\Weather-Prediction-and-Crop-Recommendation-System-\soil_suitability\outputs\soil_features_for_db_final_v2.csv"
)

OUTPUT_PATH = Path(__file__).parent / "data" / "soil_points.csv"

MODEL_CLUSTER_MAP = {
    "kmeans": "cluster_kmeans",
    "agglomerative": "cluster_agg",
    "gmm": "cluster_gmm"
}

def main():
    df = pd.read_csv(CSV_PATH)
    records = []

    for _, row in df.iterrows():
        for model, cluster_col in MODEL_CLUSTER_MAP.items():
            records.append({
                "model":          model,
                "lat":            row["lat"],
                "lon":            row["lon"],
                "taw":            row["taw"],
                "organic_carbon": row["organic_carbon"],
                "cec":            row["cec"],
                "ph":             row["ph"],
                "sand_pct":       row["sand_pct"],
                "bulk_density":   row["bulk_density"],
                "awc":            row["awc"],
                "texture_class":  row["texture_class"],
                "cluster":        int(row[cluster_col])
            })

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(OUTPUT_PATH, index=False)
    print(f"Done. {len(records)} records written to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()