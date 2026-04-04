import pytest
from app import app

@pytest.fixture
def client():
    """Create a test client for the Flask app."""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

        @pytest.fixture(autouse=True)
        def mock_model_loading(monkeypatch):
            """Mock the model and preprocessor loading to avoid reading actual files."""
            import joblib
            import numpy as np

            # Create dummy preprocessor and model
            class DummyPreprocessor:
                def transform(self, X):
                    return np.ones((X.shape[0], 10))  # dummy array

            class DummyModel:
                def predict_proba(self, X):
                    return np.array([[0.2, 0.8]] * X.shape[0])

                def predict(self, X):
                    return np.array([1] * X.shape[0])

            def mock_load(path):
                if 'preprocessor' in path:
                    return DummyPreprocessor()
                elif 'model' in path:
                    return DummyModel()
                else:
                    return None

            monkeypatch.setattr(joblib, 'load', mock_load)