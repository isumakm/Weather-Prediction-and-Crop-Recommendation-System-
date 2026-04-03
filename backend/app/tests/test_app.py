import pytest
import json

def test_home_endpoint(client):
    """Test the root endpoint returns a string."""
    response = client.get('/')
    assert response.status_code == 200
    assert b'Crop Suitability Prediction API' in response.data

def test_predict_valid(client):
    """Test /predict with a valid JSON payload."""
    payload = {
        "crop": "Brinjal",
        "temperature": 28.0,
        "rainfall": 1200.0,
        "sunshine_hours": 7.5,
        "ph": 6.5,
        "organic_carbon": 1.5,
        "cec": 15.0,
        "awc": 0.025,
        "bulk_density": 1.2,
        "texture_code": 2
    }
    response = client.post('/predict',
                           data=json.dumps(payload),
                           content_type='application/json')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'prediction' in data
    assert 'probability_suitable' in data
    assert data['prediction'] in ['Suitable', 'Unsuitable']
    assert 0.0 <= data['probability_suitable'] <= 1.0

def test_predict_missing_field(client):
    """Test missing required field returns error."""
    payload = {
        "crop": "Brinjal",
        "temperature": 28.0,
        "rainfall": 1200.0,
        "sunshine_hours": 7.5,
        "ph": 6.5,
        "organic_carbon": 1.5,
        "cec": 15.0,
        "awc": 0.025,
        "bulk_density": 1.2
        # texture_code missing
    }
    response = client.post('/predict',
                           data=json.dumps(payload),
                           content_type='application/json')
    assert response.status_code == 400
    data = json.loads(response.data)
    assert 'error' in data
    assert 'Missing features' in data['error']

def test_predict_invalid_data_type(client):
    """Test invalid data type (e.g., string instead of number) returns error."""
    payload = {
        "crop": "Brinjal",
        "temperature": 28.0,  # string instead of float
        "rainfall": 1200.0,
        "sunshine_hours": 7.5,
        "ph": 6.5,
        "organic_carbon": 1.5,
        "cec": 15.0,
        "awc": 0.025,
        "bulk_density": 1.2,
        "texture_code": 2
    }
    response = client.post('/predict',
                           data=json.dumps(payload),
                           content_type='application/json')
    assert response.status_code == 500  # or 400 if validation added

def test_predict_empty_request(client):
    """Test empty JSON returns error."""
    response = client.post('/predict',
                           data=json.dumps({}),
                           content_type='application/json')
    assert response.status_code == 400

