import random
from typing import Dict, Any

class WeatherModel:
    """
    Mock service simulating Thirusha's Weather Prediction model.
    It takes a location and date and predicts weather parameters.
    """
    def predict(self, location: str, date: str) -> Dict[str, Any]:
        # In a real scenario, this would load Thirusha's model and run inference.
        # Returning mock data for now.
        return {
            "temperature": round(random.uniform(20.0, 35.0), 2),
            "humidity": round(random.uniform(50.0, 90.0), 2),
            "rainfall": round(random.uniform(0.0, 20.0), 2)
        }

weather_service = WeatherModel()
