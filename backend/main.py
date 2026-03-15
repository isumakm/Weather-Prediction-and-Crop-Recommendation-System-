from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Any, Dict

# Import models
from models.weather_model import weather_service
from models.soil_model import soil_service
from models.ranking_model import ranking_service

app = FastAPI(title="Multi-Crop Ranking API")

# Setup CORS to allow frontend to communicate with backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins, but would be restricted in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class RecommendationRequest(BaseModel):
    location: str
    date: str
    crop: Optional[str] = None

@app.get("/")
def read_root():
    return {"message": "Welcome to the Multi-Crop Ranking System Backend!"}

@app.post("/api/recommend")
def get_recommendation(request: RecommendationRequest):
    try:
        # Step 1: Predict Weather and Soil data
        weather_data = weather_service.predict(request.location, request.date)
        soil_data = soil_service.predict(request.location, request.date)
        
        # Step 2: Determine if this is a Single Crop Analysis or Multi-Crop Ranking
        if request.crop:
            # Single Crop Analysis
            result = ranking_service.get_single_crop_analysis(
                request.crop, 
                weather_data, 
                soil_data
            )
            return {"type": "single_crop", "data": result}
        else:
            # Multi-Crop Ranking
            result = ranking_service.get_multi_crop_ranking(
                weather_data,
                soil_data
            )
            return {"type": "multi_crop", "data": result}
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
