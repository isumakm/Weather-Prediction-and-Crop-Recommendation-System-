@echo off
echo ============================================================
echo  Weather Prediction and Crop Recommendation System
echo ============================================================
echo.

:: ---- SOIL API (Port 5001) --------------------------------------------------------------------------
echo [1/4] Starting Soil API on port 5001...
start "Soil API (5001)" cmd /c "cd /d %~dp0 && python -m soil_backend.soil_app"

:wait_soil
timeout /t 2 /nobreak > nul
curl -s http://localhost:5001/soil?lat=6.9^&lon=79.9^&model=kmeans > nul 2>&1
if errorlevel 1 (
    echo     Waiting for Soil API...
    goto wait_soil
)
echo     [OK] Soil API is ready.
echo.

:: ---- WEATHER API (Port 5002) --------------------------------------------------------------------
echo [2/4] Starting Weather API on port 5002...
start "Weather API (5002)" cmd /c "cd /d %~dp0 && python weather_backend/weather_app.py"

:wait_weather
timeout /t 2 /nobreak > nul
curl -s http://localhost:5002/ > nul 2>&1
if errorlevel 1 (
    echo     Waiting for Weather API...
    goto wait_weather
)
echo     [OK] Weather API is ready.
echo.

:: ---- CROP API (Port 5003) --------------------------------------------------------------------------
echo [3/4] Starting Crop API on port 5003...
start "Crop API (5003)" cmd /c "cd /d %~dp0 && python -m crop_backend.crop_app"

:wait_crop
timeout /t 2 /nobreak > nul
curl -s http://localhost:5003/ > nul 2>&1
if errorlevel 1 (
    echo     Waiting for Crop API...
    goto wait_crop
)
echo     [OK] Crop API is ready.
echo.

:: ---- FRONTEND (Port 8080) --------------------------------------------------------------------------
echo [4/4] Starting Frontend on port 8080...
start "Frontend (8080)" cmd /c "cd /d %~dp0\frontend && python -m http.server 8080"

:wait_frontend
timeout /t 2 /nobreak > nul
curl -s http://localhost:8080/landing.html > nul 2>&1
if errorlevel 1 (
    echo     Waiting for Frontend...
    goto wait_frontend
)
echo     [OK] Frontend is ready.
echo.

:: ---- ALL READY --------------------------------------------------------------------------------------------------
echo ============================================================
echo  All services are up and running!
echo.
echo   Soil API    ^>  http://localhost:5001
echo   Weather API ^>  http://localhost:5002
echo   Crop API    ^>  http://localhost:5003
echo   Frontend    ^>  http://localhost:8080
echo ============================================================
echo.
echo Opening browser...
start http://localhost:8080/landing.html
echo.
echo Run stop.bat to shut down all services.