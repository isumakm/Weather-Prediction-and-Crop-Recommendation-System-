from flask import Blueprint, request, jsonify
from backend.soil_service import (
    get_soil_by_location,
    get_cluster_explanation,
    get_cluster_means
)

soil_bp = Blueprint("soil", __name__)

VALID_MODELS = {"kmeans", "agglomerative", "gmm"}


@soil_bp.route("/soil", methods=["GET"])
def soil_info():
    lat = request.args.get("lat", type=float)
    lon = request.args.get("lon", type=float)
    model = request.args.get("model", type=str)

    if lat is None or lon is None or model is None:
        return jsonify({
            "error": "lat, lon, and model are required"
        }), 400

    if model not in VALID_MODELS:
        return jsonify({
            "error": f"Invalid model. Choose one of {list(VALID_MODELS)}"
        }), 400

    soil = get_soil_by_location(lat, lon, model)

    if not soil:
        return jsonify({
            "error": "No soil data found for given location and model"
        }), 404

    explanation = get_cluster_explanation(soil["cluster"], model)
    cluster_means = get_cluster_means(soil["cluster"], model)

    return jsonify({
        "model": model,
        "soil_properties": soil,
        "cluster_explanation": explanation,
        "cluster_means": cluster_means
    })