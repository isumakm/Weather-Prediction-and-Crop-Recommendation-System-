import random
from typing import Dict, Any

class SoilModel:
    """
    Mock service simulating Chamatka's Soil Suitability model.
    It takes a location and date (and maybe some historical soil data) 
    and predicts soil parameters.
    """
    def predict(self, location: str, date: str) -> Dict[str, Any]:
        # In a real scenario, this would load Chamatka's model and run inference.
        return {
            "ph": round(random.uniform(5.5, 8.0), 2),
            "nitrogen": round(random.uniform(20.0, 100.0), 2),
            "phosphorus": round(random.uniform(10.0, 60.0), 2),
            "potassium": round(random.uniform(10.0, 80.0), 2)
        }

soil_service = SoilModel()
