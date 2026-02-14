from flask import Blueprint, request, jsonify
from backend.services.soil_service import (
    get_soil_by_location,
    get_cluster_explanation
)

soil_bp = Blueprint("soil", __name__)

@soil_bp.route("/soil", methods=["GET"])
def soil_info():
    lat = request.args.get("lat", type=float)
    lon = request.args.get("lon", type=float)

    if lat is None or lon is None:
        return jsonify({"error": "lat and lon required"}), 400

    soil = get_soil_by_location(lat, lon)

    if not soil:
        return jsonify({"error": "No soil data found"}), 404

    explanation = get_cluster_explanation(soil["cluster"])

    return jsonify({
        "soil_properties": soil,
        "cluster_explanation": explanation
    })
