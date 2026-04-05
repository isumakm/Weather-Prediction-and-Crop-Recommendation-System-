import pytest
import json
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock, mock_open, call
from pathlib import Path


# ==========================================================================
# MOCK DATA
# ==========================================================================

MOCK_SOIL_POINTS = pd.DataFrame([
    {"model": "kmeans",        "lat": 6.95, "lon": 80.19, "taw": 8.83,
     "organic_carbon": 2.91,  "cec": 16.64, "ph": 5.19,
     "sand_pct": 70.33,       "bulk_density": 1.21, "awc": 0.059,
     "texture_class": "Sandy loam", "cluster": 0},
    {"model": "kmeans",        "lat": 7.10, "lon": 80.05, "taw": 6.90,
     "organic_carbon": 3.18,  "cec": 9.77,  "ph": 4.86,
     "sand_pct": 66.27,       "bulk_density": 1.30, "awc": 0.046,
     "texture_class": "Sandy loam", "cluster": 1},
    {"model": "kmeans",        "lat": 6.70, "lon": 79.95, "taw": 6.63,
     "organic_carbon": 3.31,  "cec": 17.20, "ph": 5.34,
     "sand_pct": 78.76,       "bulk_density": 1.18, "awc": 0.044,
     "texture_class": "Loamy sand", "cluster": 2},
    {"model": "kmeans",        "lat": 7.00, "lon": 80.10, "taw": 7.79,
     "organic_carbon": 4.11,  "cec": 18.65, "ph": 4.81,
     "sand_pct": 62.41,       "bulk_density": 1.15, "awc": 0.052,
     "texture_class": "Sandy clay loam", "cluster": 3},
    {"model": "kmeans",        "lat": 6.9502, "lon": 80.1902, "taw": 8.50,
     "organic_carbon": 2.80,  "cec": 15.90, "ph": 5.10,
     "sand_pct": 69.00,       "bulk_density": 1.22, "awc": 0.057,
     "texture_class": "Sandy loam", "cluster": 0},
    {"model": "agglomerative", "lat": 6.95, "lon": 80.19, "taw": 8.83,
     "organic_carbon": 2.91,  "cec": 16.64, "ph": 5.19,
     "sand_pct": 70.33,       "bulk_density": 1.21, "awc": 0.059,
     "texture_class": "Sandy loam", "cluster": 1},
    {"model": "agglomerative", "lat": 7.10, "lon": 80.05, "taw": 6.90,
     "organic_carbon": 3.18,  "cec": 9.77,  "ph": 4.86,
     "sand_pct": 66.27,       "bulk_density": 1.30, "awc": 0.046,
     "texture_class": "Sandy loam", "cluster": 0},
    {"model": "gmm",           "lat": 6.95, "lon": 80.19, "taw": 8.83,
     "organic_carbon": 2.91,  "cec": 16.64, "ph": 5.19,
     "sand_pct": 70.33,       "bulk_density": 1.21, "awc": 0.059,
     "texture_class": "Sandy loam", "cluster": 0},
    {"model": "gmm",           "lat": 7.10, "lon": 80.05, "taw": 6.90,
     "organic_carbon": 3.18,  "cec": 9.77,  "ph": 4.86,
     "sand_pct": 66.27,       "bulk_density": 1.30, "awc": 0.046,
     "texture_class": "Sandy loam", "cluster": 1},
])

MOCK_CLUSTER_MEANS = {
    "kmeans": {
        "taw":            {"0": 8.825, "1": 6.895, "2": 6.630, "3": 7.790},
        "organic_carbon": {"0": 2.906, "1": 3.179, "2": 3.310, "3": 4.110},
        "cec":            {"0": 16.64, "1": 9.769, "2": 17.20, "3": 18.65},
        "ph":             {"0": 5.189, "1": 4.864, "2": 5.340, "3": 4.810},
        "sand_pct":       {"0": 70.328,"1": 66.245,"2": 78.760,"3": 62.410},
        "bulk_density":   {"0": 1.210, "1": 1.300, "2": 1.180, "3": 1.150},
    },
    "agglomerative": {
        "taw":            {"0": 6.895, "1": 8.825},
        "organic_carbon": {"0": 3.179, "1": 2.906},
        "cec":            {"0": 9.769, "1": 16.64},
        "ph":             {"0": 4.864, "1": 5.189},
        "sand_pct":       {"0": 66.245,"1": 70.328},
        "bulk_density":   {"0": 1.300, "1": 1.210},
    },
    "gmm": {
        "taw":            {"0": 8.825, "1": 6.895},
        "organic_carbon": {"0": 2.906, "1": 3.179},
        "cec":            {"0": 16.64, "1": 9.769},
        "ph":             {"0": 5.189, "1": 4.864},
        "sand_pct":       {"0": 70.328,"1": 66.245},
        "bulk_density":   {"0": 1.210, "1": 1.300},
    },
}

MOCK_CLUSTER_EXPLANATIONS = [
    {"model": "kmeans",        "cluster": 0, "zone_name": "Moisture Stable Low Organic Soil",
     "water_behavior": "High retention.", "nutrient_strength": "Low organic.", "acidity": "Moderate."},
    {"model": "kmeans",        "cluster": 1, "zone_name": "Nutrient Poor Moderately Acidic Soil",
     "water_behavior": "Low retention.",  "nutrient_strength": "Nutrient-poor.",  "acidity": "Moderately acidic."},
    {"model": "kmeans",        "cluster": 2, "zone_name": "Sandy Droughty Low Acidity Soil",
     "water_behavior": "Very low.",       "nutrient_strength": "Low nutrients.",  "acidity": "Low acidity."},
    {"model": "kmeans",        "cluster": 3, "zone_name": "Organic Rich Highly Acidic Soil",
     "water_behavior": "Moderate.",       "nutrient_strength": "High organic.",   "acidity": "Highly acidic."},
    {"model": "agglomerative", "cluster": 0, "zone_name": "Nutrient Storage Weakest Soil",
     "water_behavior": "Below average.",  "nutrient_strength": "Weakest.",        "acidity": "Second most acidic."},
    {"model": "agglomerative", "cluster": 1, "zone_name": "Moisture Stable Moderately Fertile Soil",
     "water_behavior": "Highest.",        "nutrient_strength": "Low organic.",    "acidity": "Moderate."},
    {"model": "agglomerative", "cluster": 2, "zone_name": "Organic Rich Highly Acidic Soil",
     "water_behavior": "Above average.",  "nutrient_strength": "Most fertile.",   "acidity": "Most acidic."},
    {"model": "agglomerative", "cluster": 3, "zone_name": "Sandy Droughty Low Acidity Soil",
     "water_behavior": "Lowest.",         "nutrient_strength": "Moderate.",       "acidity": "Least acidic."},
    {"model": "gmm",           "cluster": 0, "zone_name": "Moisture Stable Low Organic Soil",
     "water_behavior": "High retention.", "nutrient_strength": "Low organic.",    "acidity": "Moderate."},
    {"model": "gmm",           "cluster": 1, "zone_name": "Nutrient Poor Moderately Acidic Soil",
     "water_behavior": "Low retention.",  "nutrient_strength": "Nutrient-poor.",  "acidity": "Moderately acidic."},
    {"model": "gmm",           "cluster": 2, "zone_name": "Sandy Droughty Low Acidity Soil",
     "water_behavior": "Very low.",       "nutrient_strength": "Low nutrients.",  "acidity": "Low acidity."},
    {"model": "gmm",           "cluster": 3, "zone_name": "Organic Rich Highly Acidic Soil",
     "water_behavior": "Moderate.",       "nutrient_strength": "High organic.",   "acidity": "Highly acidic."},
]


# ==========================================================================
# FIXTURES
# ==========================================================================

@pytest.fixture(autouse=True)
def mock_service_data():
    with patch("backend.soil_service._SOIL_POINTS",           MOCK_SOIL_POINTS), \
         patch("backend.soil_service._CLUSTER_MEANS",         MOCK_CLUSTER_MEANS), \
         patch("backend.soil_service._CLUSTER_EXPLANATIONS",  MOCK_CLUSTER_EXPLANATIONS):
        yield


@pytest.fixture
def client():
    from backend.soil_app import app
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


@pytest.fixture
def sample_input_df():
    np.random.seed(42)
    n = 10
    return pd.DataFrame({
        "lat":            np.random.uniform(6.5,  7.2,  n),
        "lon":            np.random.uniform(79.8, 80.2, n),
        "taw":            np.random.uniform(5,    12,   n),
        "organic_carbon": np.random.uniform(2,    5,    n),
        "cec":            np.random.uniform(8,    20,   n),
        "ph":             np.random.uniform(4.5,  6.0,  n),
        "sand_pct":       np.random.uniform(50,   90,   n),
        "bulk_density":   np.random.uniform(1.1,  1.5,  n),
        "awc":            np.random.uniform(0.03, 0.08, n),
        "texture_class":  ["Sandy loam"] * n,
        "cluster_kmeans": np.random.randint(0, 4, n),
        "cluster_agg":    np.random.randint(0, 4, n),
        "cluster_gmm":    np.random.randint(0, 4, n),
    })


# ==========================================================================
# SECTION 1 - app.py
# ==========================================================================

def test_app_is_created():
    from backend.soil_app import app
    assert app is not None


def test_cors_enabled_on_valid_request(client):
    response = client.get(
        "/soil?lat=6.95&lon=80.19&model=kmeans",
        headers={"Origin": "http://localhost"}
    )
    assert "Access-Control-Allow-Origin" in response.headers


def test_cors_enabled_on_error_response(client):
    response = client.get(
        "/soil?lat=6.95&lon=80.19",
        headers={"Origin": "http://localhost"}
    )
    assert "Access-Control-Allow-Origin" in response.headers


def test_blueprint_registered():
    from backend.soil_app import app
    assert "soil" in app.blueprints


def test_soil_route_reachable_via_blueprint(client):
    assert client.get("/soil?lat=6.95&lon=80.19&model=kmeans").status_code != 404


def test_app_run_is_callable():
    from backend.soil_app import app
    import inspect
    sig = inspect.signature(app.run)
    assert "debug" in sig.parameters


# ==========================================================================
# SECTION 2 - soil_routes.py :: HTTP methods
# ==========================================================================

def test_get_method_allowed(client):
    assert client.get("/soil?lat=6.95&lon=80.19&model=kmeans").status_code != 405


def test_post_method_not_allowed(client):
    assert client.post("/soil").status_code == 405


def test_put_method_not_allowed(client):
    assert client.put("/soil").status_code == 405


def test_delete_method_not_allowed(client):
    assert client.delete("/soil").status_code == 405


def test_patch_method_not_allowed(client):
    assert client.patch("/soil").status_code == 405


# ==========================================================================
# SECTION 3 - soil_routes.py :: Content-Type
# ==========================================================================

def test_success_content_type_is_json(client):
    assert "application/json" in client.get("/soil?lat=6.95&lon=80.19&model=kmeans").content_type


def test_400_content_type_is_json(client):
    assert "application/json" in client.get("/soil?lat=6.95&lon=80.19").content_type


def test_404_content_type_is_json(client):
    assert "application/json" in client.get("/soil?lat=8.00&lon=80.00&model=kmeans").content_type


# ==========================================================================
# SECTION 4 - soil_routes.py :: 400 - missing / invalid parameters
# ==========================================================================

def test_missing_lat_returns_400(client):
    assert client.get("/soil?lon=80.19&model=kmeans").status_code == 400


def test_missing_lon_returns_400(client):
    assert client.get("/soil?lat=6.95&model=kmeans").status_code == 400


def test_missing_model_returns_400(client):
    assert client.get("/soil?lat=6.95&lon=80.19").status_code == 400


def test_all_params_missing_returns_400(client):
    assert client.get("/soil").status_code == 400


def test_missing_lat_and_lon_returns_400(client):
    assert client.get("/soil?model=kmeans").status_code == 400


def test_missing_lat_and_model_returns_400(client):
    assert client.get("/soil?lon=80.19").status_code == 400


def test_missing_lon_and_model_returns_400(client):
    assert client.get("/soil?lat=6.95").status_code == 400


def test_missing_params_exact_error_message(client):
    data = json.loads(client.get("/soil").data)
    assert data["error"] == "lat, lon, and model are required"


def test_non_numeric_lat_returns_400(client):
    assert client.get("/soil?lat=abc&lon=80.19&model=kmeans").status_code == 400


def test_non_numeric_lon_returns_400(client):
    assert client.get("/soil?lat=6.95&lon=abc&model=kmeans").status_code == 400


def test_invalid_model_returns_400(client):
    assert client.get("/soil?lat=6.95&lon=80.19&model=spectral").status_code == 400


def test_invalid_model_exact_error_message(client):
    data = json.loads(client.get("/soil?lat=6.95&lon=80.19&model=spectral").data)
    assert "Invalid model" in data["error"]


def test_invalid_model_uppercase_kmeans_returns_400(client):
    assert client.get("/soil?lat=6.95&lon=80.19&model=KMeans").status_code == 400


def test_invalid_model_uppercase_gmm_returns_400(client):
    assert client.get("/soil?lat=6.95&lon=80.19&model=GMM").status_code == 400


def test_invalid_model_uppercase_agglomerative_returns_400(client):
    assert client.get("/soil?lat=6.95&lon=80.19&model=Agglomerative").status_code == 400


def test_invalid_model_random_string_returns_400(client):
    assert client.get("/soil?lat=6.95&lon=80.19&model=randomforest").status_code == 400


# ==========================================================================
# SECTION 5 - soil_routes.py :: VALID_MODELS set
# ==========================================================================

def test_valid_models_contains_kmeans(client):
    assert client.get("/soil?lat=6.95&lon=80.19&model=kmeans").status_code != 400


def test_valid_models_contains_agglomerative(client):
    assert client.get("/soil?lat=6.95&lon=80.19&model=agglomerative").status_code != 400


def test_valid_models_contains_gmm(client):
    assert client.get("/soil?lat=6.95&lon=80.19&model=gmm").status_code != 400


def test_valid_models_rejects_spectral(client):
    assert client.get("/soil?lat=6.95&lon=80.19&model=spectral").status_code == 400


def test_valid_models_rejects_dbscan(client):
    assert client.get("/soil?lat=6.95&lon=80.19&model=dbscan").status_code == 400


# ==========================================================================
# SECTION 6 - soil_routes.py :: 404 - location not found
# ==========================================================================

def test_out_of_bounds_lat_high_returns_404(client):
    assert client.get("/soil?lat=8.00&lon=80.00&model=kmeans").status_code == 404


def test_out_of_bounds_lat_low_returns_404(client):
    assert client.get("/soil?lat=5.00&lon=80.00&model=kmeans").status_code == 404


def test_out_of_bounds_lon_high_returns_404(client):
    assert client.get("/soil?lat=6.90&lon=81.50&model=kmeans").status_code == 404


def test_out_of_bounds_lon_low_returns_404(client):
    assert client.get("/soil?lat=6.90&lon=79.00&model=kmeans").status_code == 404


def test_no_nearby_point_returns_404(client):
    assert client.get("/soil?lat=6.50&lon=79.80&model=kmeans").status_code == 404


def test_404_exact_error_message(client):
    data = json.loads(client.get("/soil?lat=8.00&lon=80.00&model=kmeans").data)
    assert data["error"] == "No soil data found for given location and model"


# ==========================================================================
# SECTION 7 - soil_routes.py :: 200 - response structure
# ==========================================================================

def test_kmeans_returns_200(client):
    assert client.get("/soil?lat=6.95&lon=80.19&model=kmeans").status_code == 200


def test_agglomerative_returns_200(client):
    assert client.get("/soil?lat=6.95&lon=80.19&model=agglomerative").status_code == 200


def test_gmm_returns_200(client):
    assert client.get("/soil?lat=6.95&lon=80.19&model=gmm").status_code == 200


def test_response_has_model_key(client):
    data = json.loads(client.get("/soil?lat=6.95&lon=80.19&model=kmeans").data)
    assert "model" in data


def test_response_has_soil_properties_key(client):
    data = json.loads(client.get("/soil?lat=6.95&lon=80.19&model=kmeans").data)
    assert "soil_properties" in data


def test_response_has_cluster_explanation_key(client):
    data = json.loads(client.get("/soil?lat=6.95&lon=80.19&model=kmeans").data)
    assert "cluster_explanation" in data


def test_response_has_cluster_means_key(client):
    data = json.loads(client.get("/soil?lat=6.95&lon=80.19&model=kmeans").data)
    assert "cluster_means" in data


def test_response_model_matches_kmeans(client):
    assert json.loads(client.get("/soil?lat=6.95&lon=80.19&model=kmeans").data)["model"] == "kmeans"


def test_response_model_matches_agglomerative(client):
    assert json.loads(client.get("/soil?lat=6.95&lon=80.19&model=agglomerative").data)["model"] == "agglomerative"


def test_response_model_matches_gmm(client):
    assert json.loads(client.get("/soil?lat=6.95&lon=80.19&model=gmm").data)["model"] == "gmm"


def test_response_is_valid_json(client):
    response = client.get("/soil?lat=6.95&lon=80.19&model=kmeans")
    try:
        json.loads(response.data)
    except json.JSONDecodeError:
        pytest.fail("Response is not valid JSON")


def test_cluster_explanation_is_null_when_service_returns_none(client):
    with patch("backend.soil_routes.get_cluster_explanation", return_value=None):
        data = json.loads(client.get("/soil?lat=6.95&lon=80.19&model=kmeans").data)
        assert data["cluster_explanation"] is None


def test_cluster_means_is_null_when_service_returns_none(client):
    with patch("backend.soil_routes.get_cluster_means", return_value=None):
        data = json.loads(client.get("/soil?lat=6.95&lon=80.19&model=kmeans").data)
        assert data["cluster_means"] is None


# ==========================================================================
# SECTION 8 - soil_routes.py :: soil_properties fields
# ==========================================================================

def test_soil_properties_has_taw(client):
    data = json.loads(client.get("/soil?lat=6.95&lon=80.19&model=kmeans").data)
    assert "taw" in data["soil_properties"]


def test_soil_properties_has_organic_carbon(client):
    data = json.loads(client.get("/soil?lat=6.95&lon=80.19&model=kmeans").data)
    assert "organic_carbon" in data["soil_properties"]


def test_soil_properties_has_cec(client):
    data = json.loads(client.get("/soil?lat=6.95&lon=80.19&model=kmeans").data)
    assert "cec" in data["soil_properties"]


def test_soil_properties_has_ph(client):
    data = json.loads(client.get("/soil?lat=6.95&lon=80.19&model=kmeans").data)
    assert "ph" in data["soil_properties"]


def test_soil_properties_has_sand_pct(client):
    data = json.loads(client.get("/soil?lat=6.95&lon=80.19&model=kmeans").data)
    assert "sand_pct" in data["soil_properties"]


def test_soil_properties_has_bulk_density(client):
    data = json.loads(client.get("/soil?lat=6.95&lon=80.19&model=kmeans").data)
    assert "bulk_density" in data["soil_properties"]


def test_soil_properties_has_awc(client):
    data = json.loads(client.get("/soil?lat=6.95&lon=80.19&model=kmeans").data)
    assert "awc" in data["soil_properties"]


def test_soil_properties_has_texture_class(client):
    data = json.loads(client.get("/soil?lat=6.95&lon=80.19&model=kmeans").data)
    assert "texture_class" in data["soil_properties"]


def test_soil_properties_has_cluster(client):
    data = json.loads(client.get("/soil?lat=6.95&lon=80.19&model=kmeans").data)
    assert "cluster" in data["soil_properties"]


def test_soil_properties_has_lat(client):
    data = json.loads(client.get("/soil?lat=6.95&lon=80.19&model=kmeans").data)
    assert "lat" in data["soil_properties"]


def test_soil_properties_has_lon(client):
    data = json.loads(client.get("/soil?lat=6.95&lon=80.19&model=kmeans").data)
    assert "lon" in data["soil_properties"]


def test_soil_properties_no_dist_column_leaked(client):
    data = json.loads(client.get("/soil?lat=6.95&lon=80.19&model=kmeans").data)
    assert "dist" not in data["soil_properties"]


def test_soil_properties_cluster_is_integer(client):
    data = json.loads(client.get("/soil?lat=6.95&lon=80.19&model=kmeans").data)
    assert isinstance(data["soil_properties"]["cluster"], int)


def test_soil_properties_taw_is_numeric(client):
    data = json.loads(client.get("/soil?lat=6.95&lon=80.19&model=kmeans").data)
    assert isinstance(data["soil_properties"]["taw"], (int, float))


def test_soil_properties_ph_within_valid_range(client):
    data = json.loads(client.get("/soil?lat=6.95&lon=80.19&model=kmeans").data)
    assert 3.5 <= data["soil_properties"]["ph"] <= 8.5


def test_soil_properties_sand_pct_within_valid_range(client):
    data = json.loads(client.get("/soil?lat=6.95&lon=80.19&model=kmeans").data)
    assert 0 <= data["soil_properties"]["sand_pct"] <= 100


def test_soil_properties_taw_positive(client):
    data = json.loads(client.get("/soil?lat=6.95&lon=80.19&model=kmeans").data)
    assert data["soil_properties"]["taw"] > 0


def test_soil_properties_bulk_density_positive(client):
    data = json.loads(client.get("/soil?lat=6.95&lon=80.19&model=kmeans").data)
    assert data["soil_properties"]["bulk_density"] > 0


def test_soil_properties_awc_positive(client):
    data = json.loads(client.get("/soil?lat=6.95&lon=80.19&model=kmeans").data)
    assert data["soil_properties"]["awc"] > 0


# ==========================================================================
# SECTION 9 - soil_routes.py :: cluster_explanation fields
# ==========================================================================

def test_cluster_explanation_has_zone_name(client):
    data = json.loads(client.get("/soil?lat=6.95&lon=80.19&model=kmeans").data)
    assert "zone_name" in data["cluster_explanation"]


def test_cluster_explanation_has_water_behavior(client):
    data = json.loads(client.get("/soil?lat=6.95&lon=80.19&model=kmeans").data)
    assert "water_behavior" in data["cluster_explanation"]


def test_cluster_explanation_has_nutrient_strength(client):
    data = json.loads(client.get("/soil?lat=6.95&lon=80.19&model=kmeans").data)
    assert "nutrient_strength" in data["cluster_explanation"]


def test_cluster_explanation_has_acidity(client):
    data = json.loads(client.get("/soil?lat=6.95&lon=80.19&model=kmeans").data)
    assert "acidity" in data["cluster_explanation"]


def test_cluster_explanation_has_model_field(client):
    data = json.loads(client.get("/soil?lat=6.95&lon=80.19&model=kmeans").data)
    assert "model" in data["cluster_explanation"]


def test_cluster_explanation_has_cluster_field(client):
    data = json.loads(client.get("/soil?lat=6.95&lon=80.19&model=kmeans").data)
    assert "cluster" in data["cluster_explanation"]


def test_cluster_explanation_model_matches_request(client):
    data = json.loads(client.get("/soil?lat=6.95&lon=80.19&model=kmeans").data)
    assert data["cluster_explanation"]["model"] == "kmeans"


def test_cluster_explanation_cluster_matches_soil_cluster(client):
    data = json.loads(client.get("/soil?lat=6.95&lon=80.19&model=kmeans").data)
    assert data["cluster_explanation"]["cluster"] == data["soil_properties"]["cluster"]


def test_cluster_explanation_zone_name_is_string(client):
    data = json.loads(client.get("/soil?lat=6.95&lon=80.19&model=kmeans").data)
    assert isinstance(data["cluster_explanation"]["zone_name"], str)


def test_cluster_explanation_zone_name_not_empty(client):
    data = json.loads(client.get("/soil?lat=6.95&lon=80.19&model=kmeans").data)
    assert len(data["cluster_explanation"]["zone_name"]) > 0


# ==========================================================================
# SECTION 10 - soil_routes.py :: cluster_means fields
# ==========================================================================

def test_cluster_means_has_taw(client):
    data = json.loads(client.get("/soil?lat=6.95&lon=80.19&model=kmeans").data)
    assert "taw" in data["cluster_means"]


def test_cluster_means_has_organic_carbon(client):
    data = json.loads(client.get("/soil?lat=6.95&lon=80.19&model=kmeans").data)
    assert "organic_carbon" in data["cluster_means"]


def test_cluster_means_has_cec(client):
    data = json.loads(client.get("/soil?lat=6.95&lon=80.19&model=kmeans").data)
    assert "cec" in data["cluster_means"]


def test_cluster_means_has_ph(client):
    data = json.loads(client.get("/soil?lat=6.95&lon=80.19&model=kmeans").data)
    assert "ph" in data["cluster_means"]


def test_cluster_means_has_sand_pct(client):
    data = json.loads(client.get("/soil?lat=6.95&lon=80.19&model=kmeans").data)
    assert "sand_pct" in data["cluster_means"]


def test_cluster_means_has_bulk_density(client):
    data = json.loads(client.get("/soil?lat=6.95&lon=80.19&model=kmeans").data)
    assert "bulk_density" in data["cluster_means"]


def test_cluster_means_values_are_numeric(client):
    data = json.loads(client.get("/soil?lat=6.95&lon=80.19&model=kmeans").data)
    for key, val in data["cluster_means"].items():
        assert isinstance(val, (int, float)), f"Non-numeric value for {key}"


# ==========================================================================
# SECTION 11 - soil_service.py :: data path constants
# ==========================================================================

def test_cluster_explanations_path_filename():
    import backend.soil_service as svc
    assert svc.CLUSTER_EXPLANATIONS_PATH.name == "cluster_explanations.json"


def test_cluster_means_path_filename():
    import backend.soil_service as svc
    assert svc.CLUSTER_MEANS_PATH.name == "cluster_means.json"


def test_data_dir_is_relative_to_service_file():
    import backend.soil_service as svc
    assert svc.DATA_DIR.name == "data"


def test_cluster_explanations_path_inside_data_dir():
    import backend.soil_service as svc
    assert svc.CLUSTER_EXPLANATIONS_PATH.parent == svc.DATA_DIR


def test_cluster_means_path_inside_data_dir():
    import backend.soil_service as svc
    assert svc.CLUSTER_MEANS_PATH.parent == svc.DATA_DIR


# ==========================================================================
# SECTION 12 - soil_service.py :: get_soil_by_location
# ==========================================================================

from backend.soil_service import (
    get_soil_by_location,
    get_cluster_explanation,
    get_cluster_means,
)

# ---- valid results -------------------------------------------------------

def test_get_soil_returns_result_for_known_point():
    assert get_soil_by_location(6.95, 80.19, "kmeans") is not None


def test_get_soil_returns_correct_cluster():
    assert get_soil_by_location(6.95, 80.19, "kmeans")["cluster"] == 0


def test_get_soil_returns_correct_model_field():
    assert get_soil_by_location(6.95, 80.19, "kmeans")["model"] == "kmeans"


def test_get_soil_returns_all_expected_fields():
    result = get_soil_by_location(6.95, 80.19, "kmeans")
    for f in ["lat", "lon", "taw", "organic_carbon", "cec", "ph",
              "sand_pct", "bulk_density", "awc", "texture_class", "cluster", "model"]:
        assert f in result, f"Missing field: {f}"


def test_get_soil_dist_column_not_in_result():
    assert "dist" not in get_soil_by_location(6.95, 80.19, "kmeans")


def test_get_soil_lat_close_to_queried():
    assert abs(get_soil_by_location(6.95, 80.19, "kmeans")["lat"] - 6.95) < 0.01


def test_get_soil_lon_close_to_queried():
    assert abs(get_soil_by_location(6.95, 80.19, "kmeans")["lon"] - 80.19) < 0.01


def test_get_soil_taw_is_float():
    assert isinstance(get_soil_by_location(6.95, 80.19, "kmeans")["taw"], float)


def test_get_soil_agglomerative_returns_result():
    result = get_soil_by_location(6.95, 80.19, "agglomerative")
    assert result is not None
    assert result["model"] == "agglomerative"


def test_get_soil_gmm_returns_result():
    result = get_soil_by_location(6.95, 80.19, "gmm")
    assert result is not None
    assert result["model"] == "gmm"


def test_get_soil_agglomerative_cluster_differs_from_kmeans():
    kmeans = get_soil_by_location(6.95, 80.19, "kmeans")
    agg    = get_soil_by_location(6.95, 80.19, "agglomerative")
    assert kmeans["cluster"] != agg["cluster"]


def test_get_soil_slightly_offset_coord_resolves():
    assert get_soil_by_location(6.9501, 80.1901, "kmeans") is not None


def test_get_soil_selects_exact_match_over_close_point():
    result = get_soil_by_location(6.95, 80.19, "kmeans")
    assert result["lat"] == pytest.approx(6.95, abs=0.001)
    assert result["lon"] == pytest.approx(80.19, abs=0.001)


def test_get_soil_second_location_returns_cluster_1():
    result = get_soil_by_location(7.10, 80.05, "kmeans")
    assert result is not None
    assert result["cluster"] == 1


# ---- exact boundary values -----------------------------------------------

def test_get_soil_exact_lat_lower_boundary_is_valid_bounds():
    result = get_soil_by_location(6.4, 80.19, "kmeans")
    assert result is None


def test_get_soil_exact_lat_upper_boundary_is_valid_bounds():
    result = get_soil_by_location(7.3, 80.19, "kmeans")
    assert result is None


def test_get_soil_exact_lon_lower_boundary_is_valid_bounds():
    result = get_soil_by_location(6.95, 79.7, "kmeans")
    assert result is None


def test_get_soil_exact_lon_upper_boundary_is_valid_bounds():
    result = get_soil_by_location(6.95, 80.3, "kmeans")
    assert result is None


# ---- out of bounds -------------------------------------------------------

def test_get_soil_lat_just_above_upper_returns_none():
    assert get_soil_by_location(7.31, 80.00, "kmeans") is None


def test_get_soil_lat_just_below_lower_returns_none():
    assert get_soil_by_location(6.39, 80.00, "kmeans") is None


def test_get_soil_lon_just_above_upper_returns_none():
    assert get_soil_by_location(6.90, 80.31, "kmeans") is None


def test_get_soil_lon_just_below_lower_returns_none():
    assert get_soil_by_location(6.90, 79.69, "kmeans") is None


def test_get_soil_both_coords_out_of_bounds_returns_none():
    assert get_soil_by_location(9.00, 82.00, "kmeans") is None


def test_get_soil_negative_coords_returns_none():
    assert get_soil_by_location(-6.95, -80.19, "kmeans") is None


# ---- no nearby point -----------------------------------------------------

def test_get_soil_within_bounds_no_nearby_point_returns_none():
    assert get_soil_by_location(6.50, 79.80, "kmeans") is None


def test_get_soil_just_outside_threshold_returns_none():
    assert get_soil_by_location(6.95 + 0.02, 80.19, "kmeans") is None


# ---- invalid model -------------------------------------------------------

def test_get_soil_invalid_model_returns_none():
    assert get_soil_by_location(6.95, 80.19, "invalid") is None


def test_get_soil_empty_model_string_returns_none():
    assert get_soil_by_location(6.95, 80.19, "") is None


def test_get_soil_uppercase_model_returns_none():
    assert get_soil_by_location(6.95, 80.19, "KMeans") is None


def test_get_soil_spectral_model_returns_none():
    assert get_soil_by_location(6.95, 80.19, "spectral") is None


# ==========================================================================
# SECTION 13 - soil_service.py :: get_cluster_means
# ==========================================================================

def test_get_cluster_means_returns_result():
    assert get_cluster_means(0, "kmeans") is not None


def test_get_cluster_means_has_all_fields():
    result = get_cluster_means(0, "kmeans")
    for f in ["taw", "organic_carbon", "cec", "ph", "sand_pct", "bulk_density"]:
        assert f in result, f"Missing field: {f}"


def test_get_cluster_means_taw_correct():
    assert get_cluster_means(0, "kmeans")["taw"] == pytest.approx(8.825)


def test_get_cluster_means_cec_correct():
    assert get_cluster_means(0, "kmeans")["cec"] == pytest.approx(16.64)


def test_get_cluster_means_cluster_1_kmeans():
    assert get_cluster_means(1, "kmeans")["taw"] == pytest.approx(6.895)


def test_get_cluster_means_cluster_2_kmeans():
    assert get_cluster_means(2, "kmeans") is not None


def test_get_cluster_means_cluster_3_kmeans():
    assert get_cluster_means(3, "kmeans") is not None


def test_get_cluster_means_agglomerative():
    assert get_cluster_means(0, "agglomerative")["taw"] == pytest.approx(6.895)


def test_get_cluster_means_gmm():
    assert get_cluster_means(0, "gmm")["taw"] == pytest.approx(8.825)


def test_get_cluster_means_values_are_numeric():
    result = get_cluster_means(0, "kmeans")
    for key, val in result.items():
        assert isinstance(val, (int, float)), f"Non-numeric: {key}"


def test_get_cluster_means_all_positive():
    result = get_cluster_means(0, "kmeans")
    for key, val in result.items():
        assert val > 0, f"Non-positive: {key}"


def test_get_cluster_means_cluster_key_is_string_converted():
    assert get_cluster_means(0, "kmeans") is not None


def test_get_cluster_means_invalid_model_returns_none():
    assert get_cluster_means(0, "invalid_model") is None


def test_get_cluster_means_empty_model_returns_none():
    assert get_cluster_means(0, "") is None


def test_get_cluster_means_invalid_cluster_returns_empty_dict():
    result = get_cluster_means(99, "kmeans")
    assert result == {}


# ==========================================================================
# SECTION 14 - soil_service.py :: get_cluster_explanation
# ==========================================================================

def test_get_cluster_explanation_returns_result():
    assert get_cluster_explanation(0, "kmeans") is not None


def test_get_cluster_explanation_has_zone_name():
    assert "zone_name" in get_cluster_explanation(0, "kmeans")


def test_get_cluster_explanation_has_water_behavior():
    assert "water_behavior" in get_cluster_explanation(0, "kmeans")


def test_get_cluster_explanation_has_nutrient_strength():
    assert "nutrient_strength" in get_cluster_explanation(0, "kmeans")


def test_get_cluster_explanation_has_acidity():
    assert "acidity" in get_cluster_explanation(0, "kmeans")


def test_get_cluster_explanation_has_model():
    assert "model" in get_cluster_explanation(0, "kmeans")


def test_get_cluster_explanation_has_cluster():
    assert "cluster" in get_cluster_explanation(0, "kmeans")


def test_get_cluster_explanation_kmeans_cluster_0():
    assert get_cluster_explanation(0, "kmeans")["zone_name"] == "Moisture Stable Low Organic Soil"


def test_get_cluster_explanation_kmeans_cluster_1():
    assert get_cluster_explanation(1, "kmeans")["zone_name"] == "Nutrient Poor Moderately Acidic Soil"


def test_get_cluster_explanation_kmeans_cluster_2():
    assert get_cluster_explanation(2, "kmeans")["zone_name"] == "Sandy Droughty Low Acidity Soil"


def test_get_cluster_explanation_kmeans_cluster_3():
    assert get_cluster_explanation(3, "kmeans")["zone_name"] == "Organic Rich Highly Acidic Soil"


def test_get_cluster_explanation_agglomerative_all_four_clusters():
    for c in range(4):
        assert get_cluster_explanation(c, "agglomerative") is not None


def test_get_cluster_explanation_gmm_all_four_clusters():
    for c in range(4):
        assert get_cluster_explanation(c, "gmm") is not None


def test_get_cluster_explanation_model_field_matches():
    assert get_cluster_explanation(0, "kmeans")["model"] == "kmeans"


def test_get_cluster_explanation_cluster_field_matches():
    assert get_cluster_explanation(2, "kmeans")["cluster"] == 2


def test_get_cluster_explanation_zone_name_is_nonempty_string():
    result = get_cluster_explanation(0, "kmeans")
    assert isinstance(result["zone_name"], str) and len(result["zone_name"]) > 0


def test_get_cluster_explanation_wrong_cluster_returns_none():
    assert get_cluster_explanation(99, "kmeans") is None


def test_get_cluster_explanation_wrong_model_returns_none():
    assert get_cluster_explanation(0, "spectral") is None


def test_get_cluster_explanation_wrong_both_returns_none():
    assert get_cluster_explanation(99, "spectral") is None


def test_get_cluster_explanation_empty_model_returns_none():
    assert get_cluster_explanation(0, "") is None


def test_get_cluster_explanation_matches_on_both_model_and_cluster():
    result = get_cluster_explanation(0, "agglomerative")
    assert result["model"] == "agglomerative"
    assert result["cluster"] == 0


# ==========================================================================
# SECTION 15 - generate_soil_points.py :: transformation logic
# ==========================================================================

MODEL_CLUSTER_MAP_GSP = {
    "kmeans":        "cluster_kmeans",
    "agglomerative": "cluster_agg",
    "gmm":           "cluster_gmm"
}

REQUIRED_OUTPUT_COLS = [
    "model", "lat", "lon", "taw", "organic_carbon", "cec", "ph",
    "sand_pct", "bulk_density", "awc", "texture_class", "cluster"
]


def run_gsp_logic(df):
    records = []
    for _, row in df.iterrows():
        for model, cluster_col in MODEL_CLUSTER_MAP_GSP.items():
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
    return pd.DataFrame(records)


def test_gsp_model_cluster_map_has_three_entries():
    assert len(MODEL_CLUSTER_MAP_GSP) == 3


def test_gsp_model_cluster_map_correct_keys():
    assert set(MODEL_CLUSTER_MAP_GSP.keys()) == {"kmeans", "agglomerative", "gmm"}


def test_gsp_model_cluster_map_kmeans_col():
    assert MODEL_CLUSTER_MAP_GSP["kmeans"] == "cluster_kmeans"


def test_gsp_model_cluster_map_agglomerative_col():
    assert MODEL_CLUSTER_MAP_GSP["agglomerative"] == "cluster_agg"


def test_gsp_model_cluster_map_gmm_col():
    assert MODEL_CLUSTER_MAP_GSP["gmm"] == "cluster_gmm"


def test_gsp_output_row_count_is_three_times_input(sample_input_df):
    assert len(run_gsp_logic(sample_input_df)) == len(sample_input_df) * 3


def test_gsp_output_has_all_required_columns(sample_input_df):
    output = run_gsp_logic(sample_input_df)
    for col in REQUIRED_OUTPUT_COLS:
        assert col in output.columns, f"Missing: {col}"


def test_gsp_model_column_contains_only_valid_models(sample_input_df):
    output = run_gsp_logic(sample_input_df)
    assert set(output["model"].unique()) == {"kmeans", "agglomerative", "gmm"}


def test_gsp_each_model_has_same_count_as_input(sample_input_df):
    output = run_gsp_logic(sample_input_df)
    for model in ["kmeans", "agglomerative", "gmm"]:
        assert len(output[output["model"] == model]) == len(sample_input_df)


def test_gsp_cluster_is_integer(sample_input_df):
    output = run_gsp_logic(sample_input_df)
    assert output["cluster"].dtype in [int, np.int64, np.int32]


def test_gsp_cluster_values_within_range(sample_input_df):
    assert run_gsp_logic(sample_input_df)["cluster"].between(0, 3).all()


def test_gsp_no_nan_values_in_output(sample_input_df):
    assert run_gsp_logic(sample_input_df).isnull().sum().sum() == 0


def test_gsp_lat_lon_preserved(sample_input_df):
    output = run_gsp_logic(sample_input_df)
    kmeans = output[output["model"] == "kmeans"].reset_index(drop=True)
    np.testing.assert_array_almost_equal(kmeans["lat"].values, sample_input_df["lat"].values)
    np.testing.assert_array_almost_equal(kmeans["lon"].values, sample_input_df["lon"].values)


def test_gsp_kmeans_cluster_from_cluster_kmeans_col(sample_input_df):
    output = run_gsp_logic(sample_input_df)
    kmeans = output[output["model"] == "kmeans"].reset_index(drop=True)
    np.testing.assert_array_equal(kmeans["cluster"].values,
                                  sample_input_df["cluster_kmeans"].astype(int).values)


def test_gsp_agglomerative_cluster_from_cluster_agg_col(sample_input_df):
    output = run_gsp_logic(sample_input_df)
    agg = output[output["model"] == "agglomerative"].reset_index(drop=True)
    np.testing.assert_array_equal(agg["cluster"].values,
                                  sample_input_df["cluster_agg"].astype(int).values)


def test_gsp_gmm_cluster_from_cluster_gmm_col(sample_input_df):
    output = run_gsp_logic(sample_input_df)
    gmm = output[output["model"] == "gmm"].reset_index(drop=True)
    np.testing.assert_array_equal(gmm["cluster"].values,
                                  sample_input_df["cluster_gmm"].astype(int).values)


def test_gsp_source_cluster_cols_not_in_output(sample_input_df):
    output = run_gsp_logic(sample_input_df)
    assert "cluster_kmeans" not in output.columns
    assert "cluster_agg"    not in output.columns
    assert "cluster_gmm"    not in output.columns


def test_gsp_output_path_ends_with_soil_points_csv():
    from backend.generate_soil_points import OUTPUT_PATH
    assert OUTPUT_PATH.name == "soil_points.csv"
    assert OUTPUT_PATH.parent.name == "data"


def test_gsp_mkdir_called_with_parents_and_exist_ok(sample_input_df, tmp_path):
    output_file = tmp_path / "data" / "soil_points.csv"
    with patch("backend.generate_soil_points.OUTPUT_PATH", output_file), \
         patch("backend.generate_soil_points.CSV_PATH"), \
         patch("pandas.read_csv", return_value=sample_input_df):
        from backend.generate_soil_points import main
        main()
    assert output_file.exists()


def test_gsp_to_csv_written_without_index(sample_input_df, tmp_path):
    output_file = tmp_path / "data" / "soil_points.csv"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with patch("backend.generate_soil_points.OUTPUT_PATH", output_file), \
         patch("backend.generate_soil_points.CSV_PATH"), \
         patch("pandas.read_csv", return_value=sample_input_df):
        from backend.generate_soil_points import main
        main()
    written = pd.read_csv(output_file)
    assert "Unnamed: 0" not in written.columns


def test_gsp_print_contains_record_count(sample_input_df, tmp_path, capsys):
    output_file = tmp_path / "data" / "soil_points.csv"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with patch("backend.generate_soil_points.OUTPUT_PATH", output_file), \
         patch("backend.generate_soil_points.CSV_PATH"), \
         patch("pandas.read_csv", return_value=sample_input_df):
        from backend.generate_soil_points import main
        main()
    captured = capsys.readouterr()
    expected_count = len(sample_input_df) * 3
    assert str(expected_count) in captured.out


def test_gsp_main_block_calls_main():
    import ast as ast_mod
    import backend.generate_soil_points as gsp_module
    src = Path(gsp_module.__file__).read_text()
    tree = ast_mod.parse(src)
    found = False
    for node in ast_mod.walk(tree):
        if not isinstance(node, ast_mod.If):
            continue
        test = node.test
        if not isinstance(test, ast_mod.Compare):
            continue
        left = test.left
        ops = test.ops
        comparators = test.comparators
        if len(ops) != 1 or not isinstance(ops[0], ast_mod.Eq):
            continue
        name_node = None
        str_node = None
        for candidate in [left] + list(comparators):
            if isinstance(candidate, ast_mod.Name) and candidate.id == "__name__":
                name_node = candidate
            if isinstance(candidate, ast_mod.Constant) and candidate.value == "__main__":
                str_node = candidate
        if name_node is None or str_node is None:
            continue
        for stmt in node.body:
            if isinstance(stmt, ast_mod.Expr) and isinstance(stmt.value, ast_mod.Call):
                call = stmt.value
                if isinstance(call.func, ast_mod.Name) and call.func.id == "main":
                    found = True
    assert found, "__main__ guard calling main() not found in generate_soil_points.py"


# ==========================================================================
# SECTION 16 - cluster_explanations.json :: structure validation
# ==========================================================================

REAL_EXPLANATIONS_PATH = Path(__file__).parent.parent / "backend" / "data" / "cluster_explanations.json"


@pytest.fixture
def real_explanations():
    if not REAL_EXPLANATIONS_PATH.exists():
        pytest.skip("cluster_explanations.json not found")
    with open(REAL_EXPLANATIONS_PATH) as f:
        return json.load(f)


def test_explanations_is_list(real_explanations):
    assert isinstance(real_explanations, list)


def test_explanations_has_twelve_entries(real_explanations):
    assert len(real_explanations) == 12


def test_explanations_each_entry_has_model(real_explanations):
    for e in real_explanations:
        assert "model" in e


def test_explanations_each_entry_has_cluster(real_explanations):
    for e in real_explanations:
        assert "cluster" in e


def test_explanations_each_entry_has_zone_name(real_explanations):
    for e in real_explanations:
        assert "zone_name" in e


def test_explanations_each_entry_has_water_behavior(real_explanations):
    for e in real_explanations:
        assert "water_behavior" in e


def test_explanations_each_entry_has_nutrient_strength(real_explanations):
    for e in real_explanations:
        assert "nutrient_strength" in e


def test_explanations_each_entry_has_acidity(real_explanations):
    for e in real_explanations:
        assert "acidity" in e


def test_explanations_only_valid_models(real_explanations):
    for e in real_explanations:
        assert e["model"] in {"kmeans", "agglomerative", "gmm"}


def test_explanations_cluster_values_within_range(real_explanations):
    for e in real_explanations:
        assert 0 <= e["cluster"] <= 3


def test_explanations_no_duplicate_model_cluster_pairs(real_explanations):
    pairs = [(e["model"], e["cluster"]) for e in real_explanations]
    assert len(pairs) == len(set(pairs))


def test_explanations_all_three_models_present(real_explanations):
    models = {e["model"] for e in real_explanations}
    assert models == {"kmeans", "agglomerative", "gmm"}


def test_explanations_kmeans_has_four_clusters(real_explanations):
    assert len([e for e in real_explanations if e["model"] == "kmeans"]) == 4


def test_explanations_agglomerative_has_four_clusters(real_explanations):
    assert len([e for e in real_explanations if e["model"] == "agglomerative"]) == 4


def test_explanations_gmm_has_four_clusters(real_explanations):
    assert len([e for e in real_explanations if e["model"] == "gmm"]) == 4


def test_explanations_zone_names_nonempty_strings(real_explanations):
    for e in real_explanations:
        assert isinstance(e["zone_name"], str) and len(e["zone_name"]) > 0


def test_explanations_all_text_fields_nonempty(real_explanations):
    for e in real_explanations:
        for field in ["water_behavior", "nutrient_strength", "acidity"]:
            assert len(e[field]) > 0

# ==========================================================================
# SECTION 17 - end-to-end integration
# ==========================================================================

def test_e2e_cluster_in_properties_matches_explanation(client):
    data = json.loads(client.get("/soil?lat=6.95&lon=80.19&model=kmeans").data)
    assert data["soil_properties"]["cluster"] == data["cluster_explanation"]["cluster"]


def test_e2e_explanation_model_matches_request(client):
    data = json.loads(client.get("/soil?lat=6.95&lon=80.19&model=kmeans").data)
    assert data["cluster_explanation"]["model"] == data["model"]


def test_e2e_cluster_key_exists_in_means(client):
    data = json.loads(client.get("/soil?lat=6.95&lon=80.19&model=kmeans").data)
    cluster_key = str(data["soil_properties"]["cluster"])
    assert cluster_key in MOCK_CLUSTER_MEANS["kmeans"]["taw"]


def test_e2e_two_locations_different_clusters(client):
    d1 = json.loads(client.get("/soil?lat=6.95&lon=80.19&model=kmeans").data)
    d2 = json.loads(client.get("/soil?lat=7.10&lon=80.05&model=kmeans").data)
    assert d1["soil_properties"]["cluster"] != d2["soil_properties"]["cluster"]


def test_e2e_all_three_models_200_same_location(client):
    for model in ["kmeans", "agglomerative", "gmm"]:
        assert client.get(f"/soil?lat=6.95&lon=80.19&model={model}").status_code == 200


def test_e2e_full_pipeline_kmeans(client):
    data = json.loads(client.get("/soil?lat=6.95&lon=80.19&model=kmeans").data)
    assert data["model"] == "kmeans"
    assert data["soil_properties"]["taw"] > 0
    assert 0 <= data["soil_properties"]["cluster"] <= 3
    assert len(data["cluster_explanation"]["zone_name"]) > 0
    assert data["cluster_means"]["taw"] > 0


def test_e2e_full_pipeline_agglomerative(client):
    data = json.loads(client.get("/soil?lat=6.95&lon=80.19&model=agglomerative").data)
    assert data["model"] == "agglomerative"
    assert data["soil_properties"]["taw"] > 0
    assert 0 <= data["soil_properties"]["cluster"] <= 3
    assert len(data["cluster_explanation"]["zone_name"]) > 0
    assert data["cluster_means"]["taw"] > 0


def test_e2e_full_pipeline_gmm(client):
    data = json.loads(client.get("/soil?lat=6.95&lon=80.19&model=gmm").data)
    assert data["model"] == "gmm"
    assert data["soil_properties"]["taw"] > 0
    assert 0 <= data["soil_properties"]["cluster"] <= 3
    assert len(data["cluster_explanation"]["zone_name"]) > 0
    assert data["cluster_means"]["taw"] > 0