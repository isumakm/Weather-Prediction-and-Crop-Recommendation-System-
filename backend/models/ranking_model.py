from typing import Dict, Any, List, Optional
import random

# A mock list of crops for the demo
MOCK_CROPS = [
    "Rice", "Maize", "Chickpea", "Kidneybeans", "Pigeonpeas", 
    "Mothbeans", "Mungbean", "Blackgram", "Lentil", "Pomegranate", 
    "Banana", "Mango", "Grapes", "Watermelon", "Muskmelon", 
    "Apple", "Orange", "Papaya", "Coconut", "Cotton", "Jute", "Coffee"
]

class RankingModel:
    """
    Mock service simulating the Multi-Crop Ranking SVM Model.
    """
    def __init__(self):
        # NOTE: Once the real SVM model is available (e.g. svm_model.pkl),
        # it should be loaded here using joblib or pickle.
        pass

    def get_multi_crop_ranking(self, weather_data: Dict[str, Any], soil_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Takes weather and soil data as input and returns a ranked list of crops.
        """
        # Feature vector could be built here from weather_data and soil_data
        
        # Mocking the ranking: Shuffle the crop list and assign random confidence scores
        shuffled_crops = random.sample(MOCK_CROPS, len(MOCK_CROPS))
        results = []
        base_score = 95.0
        for i, crop in enumerate(shuffled_crops[:10]): # Return top 10
            results.append({
                "rank": i + 1,
                "crop": crop,
                "suitability_score": round(base_score - (i * random.uniform(2.0, 5.0)), 2)
            })
        return results

    def get_single_crop_analysis(self, crop: str, weather_data: Dict[str, Any], soil_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Provides a suitability report for a specific crop based on input data.
        """
        crop_name = crop.capitalize()
        # Mock suitability analysis
        is_suitable = random.choice([True, False, True, True]) # Skewed towards True for demo
        score = random.uniform(70.0, 98.0) if is_suitable else random.uniform(30.0, 60.0)
        
        reasons = []
        if weather_data["rainfall"] < 5.0 and crop_name == "Rice":
            reasons.append("Rainfall is too low for optimal rice growth.")
        if soil_data["nitrogen"] < 30.0:
            reasons.append("Nitrogen levels are slightly below recommended parameters.")
            
        if not reasons:
            reasons.append("All parameters are within optimal ranges for this crop.")

        return {
            "crop": crop_name,
            "is_suitable": is_suitable,
            "suitability_score": round(score, 2),
            "analysis_report": reasons,
            "weather_context": weather_data,
            "soil_context": soil_data
        }

ranking_service = RankingModel()
