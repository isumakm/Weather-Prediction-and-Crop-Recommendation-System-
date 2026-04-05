@echo off
echo ============================================================
echo  Setup - Weather Prediction and Crop Recommendation System
echo ============================================================
echo.

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH.
    echo.
    echo Please install Python from https://www.python.org/downloads/
    echo IMPORTANT: During installation tick "Add Python to PATH"
    echo.
    pause
    exit /b 1
)

for /f "tokens=*" %%v in ('python --version') do echo Found: %%v
echo.

:: Install required packages
echo Installing required packages...
pip install flask flask-cors pandas scikit-learn joblib xgboost numpy
echo.

:: Check critical data files
echo Checking required files...
set MISSING=0

for %%f in (
    "soil_backend\data\soil_points.csv"
    "soil_backend\data\cluster_explanations.json"
    "soil_backend\data\cluster_means.json"
    "weather_backend\rf_seasonal_temperature.pkl"
    "weather_backend\rf_seasonal_rainfall.pkl"
    "weather_backend\rf_seasonal_sunshine.pkl"
    "crop_backend\models\preprocessor.pkl"
    "crop_backend\models\xgboost_best.pkl"
    "frontend\web_cluster_points_kmeans.csv"
    "frontend\web_cluster_points_agglomerative.csv"
    "frontend\web_cluster_points_gmm.csv"
    "frontend\web_cluster_means.json"
    "frontend\web_shap.json"
) do (
    if not exist "%~dp0%%f" (
        echo   [MISSING] %%f
        set MISSING=1
    ) else (
        echo   [OK]      %%f
    )
)

echo.
if "%MISSING%"=="1" (
    echo [WARNING] Some files are missing - see above.
    echo The system may not work correctly without them.
) else (
    echo All required files found.
)

echo.
echo ============================================================
echo  Setup complete. Run start.bat to launch the system.
echo ============================================================
echo.
pause
