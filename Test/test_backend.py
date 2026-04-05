import os
import sys
import importlib
import pytest

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class FakeTempModel:
    def predict(self, X):
        return [27.55]


class FakeRainModel:
    def predict(self, X):
        # log-scale rainfall prediction
        return [6.2166061010848646]   # expm1 -> about 500


class FakeSunModel:
    def predict(self, X):
        return [7.25]


@pytest.fixture
def client(monkeypatch):
    # Pretend model files exist
    monkeypatch.setattr("os.path.exists", lambda path: True)

    fake_models = [FakeTempModel(), FakeRainModel(), FakeSunModel()]

    def fake_load(path):
        return fake_models.pop(0)

    monkeypatch.setattr("joblib.load", fake_load)

    # Fresh import after monkeypatch
    if "app" in sys.modules:
        del sys.modules["app"]

    app_module = importlib.import_module("app")
    app_module.app.config["TESTING"] = True

    with app_module.app.test_client() as test_client:
        yield test_client


def test_home_route(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"Seasonal weather backend running" in response.data


def test_schema_route(client):
    response = client.get("/schema")
    assert response.status_code == 200

    data = response.get_json()
    assert "features" in data
    assert "example" in data
    assert data["features"] == ["season", "year", "location_id"]


def test_predict_valid_input(client):
    payload = {
        "season": "North-east monsoon",
        "year": 2026,
        "location_id": "1"
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 200

    data = response.get_json()
    assert "temperature_C" in data
    assert "rainfall_mm" in data
    assert "sunshine_h" in data

    assert isinstance(data["temperature_C"], float)
    assert isinstance(data["rainfall_mm"], float)
    assert isinstance(data["sunshine_h"], float)


def test_predict_missing_location(client):
    payload = {
        "season": "North-east monsoon",
        "year": 2026
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 400

    data = response.get_json()
    assert "error" in data


def test_predict_invalid_season(client):
    payload = {
        "season": "Wrong Season",
        "year": 2026,
        "location_id": "1"
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 400

    data = response.get_json()
    assert data["error"] == "Invalid season value"


def test_predict_with_date_auto_conversion(client):
    payload = {
        "date": "2026-01-15",
        "location_id": "1"
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 200

    data = response.get_json()
    assert data["season"] == "North-east monsoon"
    assert data["year"] == 2026


def test_predict_invalid_date(client):
    payload = {
        "date": "invalid-date",
        "location_id": "1"
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 400

    data = response.get_json()
    assert data["error"] == "Invalid date format. Use YYYY-MM-DD"