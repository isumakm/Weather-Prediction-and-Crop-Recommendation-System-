# 🌾 Weather Prediction and Crop Recommendation System
# 🌾 Weather Prediction and Crop Recommendation System - Western Province, Sri Lanka

**An integrated agricultural decision support system for the Western Province of Sri Lanka**


## 📝 Project Overview

### The Challenge: Climate Vulnerability in Sri Lanka's Western Province
### The Challenge

The Western Province (Colombo, Gampaha, and Kalutara districts) faces escalating climate challenges that threaten agricultural productivity and food security. Farmers and agricultural officers currently make planting decisions based on tradition and general knowledge rather than actual soil and weather data specific to their location. This leads to suboptimal crop selection, poor yields, and wasted resources.

- **Colombo District:** Untimely rainfall during dry months disrupts flowering cycles in fruit crops like rambutan and mangosteen. Excessive rainfall and frequent flooding pose ongoing threats.
- **Gampaha District:** Intense rains during the first inter-monsoon (April) and southwest monsoon (May-July) adversely affect fruit yields and cause flash floods in paddy cultivation areas.
- **Kalutara District:** Torrential rainfall causes regular flooding from the Kalu Ganga, particularly during the southwest monsoon and weather systems developing in the Bay of Bengal (October-January).

### Our Solution

A web-based decision support system that integrates location-specific soil zone classification, seasonal weather prediction, and machine learning-based crop suitability analysis into a single guided four-step workflow, enabling informed, data-driven planting decisions without any technical knowledge required.


## ✨ Features

- **Landing Page** - System overview, step-by-step instructions, and entry point before accessing the system.
- **User Authentication** - Sign in, sign up, or continue as guest.
- **Interactive Map** - Click any location in the Western Province to begin analysis. Includes a live location button that detects your current GPS position using exact district polygon boundaries.
- **Soil Zone Analysis** - Classifies soil into management zones.
- **Seasonal Weather Prediction** - Predicts temperature, rainfall, and sunshine hours for any season based on historical data.
- **Multi-Crop Ranking** - Ranks all 21 crops by suitability.
- **Single Crop Suitability Report** - Detailed analysis for a specific crop including probability score, limiting factors, and management advisories.
- **PDF Report Download** - Download a formatted suitability report as a PDF directly from the browser.
- **Guided Workflow** - Data flows automatically between pages. No manual re-entry at any step.


## 🧠 Machine Learning Models

### Soil Component
| Model | Purpose |
|---|---|
| K-Means Clustering | Soil zone classification |
| Agglomerative Clustering | Alternative zone classification |
| Gaussian Mixture Model (GMM) | Alternative zone classification |

All three models produce zone classifications with SHAP-based agronomic explanations covering water behaviour, nutrient strength, and acidity for each zone.

### Weather Component
| Model | Target Variable |
|---|---|
| Random Forest Regressor | Temperature (°C) |
| Random Forest Regressor | Rainfall (mm) |
| Random Forest Regressor | Sunshine hours (h/day) |


### Crop Suitability Component
| Model | Purpose |
|---|---|
| XGBoost Classifier | Single crop suitability |
| Support Vector Machine (SVM) | Multi-crop ranking |



## 📁 Project Structure

```
project/
│
├── frontend/
│   ├── landing.html                     
│   ├── index.html                        
│   ├── weather.html                     
│   ├── crop.html                        
│   ├── report.html                       
│   ├── auth.js                         
│   ├── western_province_districts.json  
│   ├── web_cluster_points_kmeans.csv    
│   ├── web_cluster_points_agglomerative.csv
│   ├── web_cluster_points_gmm.csv
│   ├── web_cluster_means.json
│   └── web_shap.json
│
├── soil_backend/
│   ├── soil_app.py                     
│   ├── soil_routes.py
│   ├── soil_service.py
│   └── data/
│       ├── soil_points.csv
│       ├── cluster_explanations.json
│       └── cluster_means.json
│
├── weather_backend/
│   ├── weather_app.py                  
│   ├── rf_seasonal_temperature.pkl
│   ├── rf_seasonal_rainfall.pkl
│   └── rf_seasonal_sunshine.pkl
│
├── crop_backend/
│   ├── crop_app.py                       
│   └── models/
│       ├── preprocessor.pkl
│       ├── xgboost_best.pkl
│       └── tuned_svm.pkl                
│
├── setup.bat
├── start.bat                             
└── stop.bat                              
```


## 🚀 Getting Started

### Prerequisites

- Python 3.11
- Windows (for `.bat` scripts) — Linux/Mac users can run each service manually

### First-time Setup

Run `setup.bat` once before starting the system for the first time:

```bat
setup.bat
```

This checks that Python is installed, installs all required packages, and verifies all model and data files are present.

### Starting the System

```bat
start.bat
```

This launches all four services sequentially with health checks, then opens the browser automatically at the landing page.

### Stopping the System

```bat
stop.bat
```

This kills all running services and closes all terminal windows.

### Manual Start (if .bat files don't work)

Open four separate terminals and run:

```bash
# Terminal 1 - Soil API
python -m soil_backend.soil_app

# Terminal 2 - Weather API
python weather_backend/weather_app.py

# Terminal 3 - Crop API
python -m crop_backend.crop_app

# Terminal 4 - Frontend
cd frontend && python -m http.server 8080
```

Then open `http://localhost:8080/landing.html` in your browser.


## 🌐 API Reference

### Soil API - `http://localhost:5001`

| Endpoint | Method | Description |
|---|---|---|
| `/soil` | GET | Get soil properties for a location |

**Parameters:** `lat` (float), `lon` (float), `model` (kmeans / agglomerative / gmm)



### Weather API - `http://localhost:5002`

| Endpoint | Method | Description |
|---|---|---|
| `/predict` | POST | Predict weather for a season and district |
| `/schema` | GET | View accepted inputs and valid seasons |

**Valid seasons:** `North-east monsoon` · `Intermonsoon after North-east monsoon` · `South-west monsoon` · `Intermonsoon after South-west monsoon`

**Location IDs:** `1` = Colombo · `2` = Gampaha · `3` = Kalutara



### Crop API - `http://localhost:5003`

| Endpoint | Method | Description |
|---|---|---|
| `/predict` | POST | Single crop suitability analysis |
| `/rank` | POST | Rank all 21 crops |




## 🌱 Supported Crops (21 total)

Banana · Bitter Gourd · Brinjal · Capsicum · Cucumber · Ginger · Kiriala · Luffa · Mangosteen · Manioc · Okra · Papaya · Passion Fruit · Pineapple · Radish · Rambutan · Snake Gourd · Sweet Potato · Turmeric · Yams · Yard Long Bean



## 🔄 Data Flow

```
User clicks map location
        ↓
Soil API returns soil properties + zone explanation
        ↓
Weather API returns temperature, rainfall, sunshine
        ↓
Crop API ranks all 21 crops OR analyses a specific crop
        ↓
User downloads PDF suitability report
```

Data is passed between pages automatically using `sessionStorage` and URL parameters. The user never needs to re-enter any information.



## 📦 Dependencies

```
flask
flask-cors
joblib
pandas
numpy
scikit-learn==1.0.2
xgboost
gunicorn
```

> **Important:** `scikit-learn` must be version `1.0.2` to match the pickle files. Using a newer version will cause the models to fail on load.

Install all at once:
```bash
pip install flask flask-cors joblib pandas numpy scikit-learn==1.0.2 xgboost gunicorn
```



## 🗺️ District Boundaries

The system uses exact polygon boundaries from the **GADM Level 1 dataset** (`gadm41_LKA_1.json`) for district detection. The three Western Province districts (Colombo, Gampaha, Kalutara) are extracted into `western_province_districts.json` and used in:

- `weather.html` - to determine which district a clicked location belongs to
- `index.html` - to check whether a live GPS location is inside the Western Province

A ray-casting point-in-polygon algorithm is used for accurate boundary detection instead of simple lat/lon bounding boxes.



## 🔒 Authentication

Authentication is handled by the shared `auth.js` file, which is loaded by every page. User accounts are stored in `localStorage`. All pages check for a valid session - if a user tries to open any page directly without going through the landing page, they are redirected.

Three access modes:
- **Sign In** - restores previous session with name displayed
- **Sign Up** - creates a new account stored in browser
- **Guest** - full access without saving a session



## 🚧 Known Limitations

- Currently covers the Western Province only - Soil data and models are not available for other provinces
- Interface is in English only - Sinhala and Tamil support not yet implemented
- Authentication is client-side only - Not suitable for production deployment
- Weather models are based on historical patterns - Accuracy may decrease with climate change over time



## 🔮 Future Enhancements

- Extend coverage to all nine Sri Lanka provinces
- Add Sinhala and Tamil language support
- Integrate real-time weather data from the Sri Lanka Meteorological Department
- Develop a mobile application for field officers in low-connectivity areas


The Western Province, covering Colombo, Gampaha, and Kalutara districts, faces distinct and escalating climate challenges that threaten agricultural productivity and food security:

* **Colombo District:** While drought is uncommon, untimely rainfall during traditionally dry months (February to mid-March) disrupts flowering cycles in valuable fruit crops like rambutan and mangosteen. Excessive rainfall and frequent flooding pose significant threats to agricultural activities throughout the region.
* **Gampaha District:** Characterized by flat terrain ideal for fruit cultivation, this district suffers from intense rains during the first inter-monsoon (April) and southwest monsoon (May to mid-July). These downpours adversely affect fruit yields, block drainage systems, and cause flash floods that impact paddy cultivation in river floodplains.
* **Kalutara District:** Heavily affected by torrential rainfall, this district's low-lying areas face regular flooding from the overflowing Kalu Ganga, particularly during the southwest monsoon and when weather systems develop in the Bay of Bengal (October to early January).
## 👥 Team

### Our Solution: An Integrated Climate Intelligence Platform
| Member | Component |
|---|---|
| Withanage Isum Akmitha | Single Crop Analysis |
| Chamathka Thamoshi Sooriyaarachchi | Soil Suitability Analysis |
| Thirusha Kannathasan | Weather Prediction |
| Manura Lavod Dulnath Hettiarachchi | Multi Crop Ranking |

This system provides data-driven, climate-resilient agricultural decision support specifically tailored to the Western Province's unique challenges. By integrating high-resolution weather forecasting, detailed soil analysis, and crop science, we help farmers and agricultural officers:

* Anticipate climate risks before they impact crops
* Select appropriate crops and varieties based on soil resilience characteristics
* Implement timely adaptations to mitigate weather-related losses
* Optimize resource allocation for sustainable productivity

## 🖧 System Architecture
## 📄 License

![alt text](https://github.com/isumakm/Weather-Prediction-and-Crop-Recommendation-System-/blob/main/System%20Architecture.jpg?raw=true)
This project was developed as part of the BSc (Hons) in Artificial Intelligence and Data Science at Robert Gordon University Aberdeen. All rights reserved.
