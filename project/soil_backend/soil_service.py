import pandas as pd
import json
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
CLUSTER_EXPLANATIONS_PATH = DATA_DIR / "cluster_explanations.json"
CLUSTER_MEANS_PATH        = DATA_DIR / "cluster_means.json"

_SOIL_POINTS = pd.read_csv(DATA_DIR / "soil_points.csv")

with open(CLUSTER_EXPLANATIONS_PATH) as f:
    _CLUSTER_EXPLANATIONS = json.load(f)

with open(CLUSTER_MEANS_PATH) as f:
    _CLUSTER_MEANS = json.load(f)


def get_soil_by_location(lat, lon, model):
    if not (6.4 <= lat <= 7.3 and 79.7 <= lon <= 80.3):
        return None

    subset = _SOIL_POINTS[_SOIL_POINTS["model"] == model].copy()
    subset["dist"] = (subset["lat"] - lat) ** 2 + (subset["lon"] - lon) ** 2

    nearest = subset[subset["dist"] <= 0.0001]
    if nearest.empty:
        return None

    row = nearest.loc[nearest["dist"].idxmin()]
    return row.drop("dist").to_dict()


def get_cluster_means(cluster_id, model):
    model_means = _CLUSTER_MEANS.get(model)
    if not model_means:
        return None

    cluster_key = str(cluster_id)
    return {
        feature: values[cluster_key]
        for feature, values in model_means.items()
        if cluster_key in values
    }


def get_cluster_explanation(cluster_id, model):
    for entry in _CLUSTER_EXPLANATIONS:
        if entry["model"] == model and entry["cluster"] == cluster_id:
            return entry
    return None