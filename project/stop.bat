@echo off
echo ============================================================
echo  Stopping all services...
echo ============================================================
echo.

setlocal enabledelayedexpansion

:: Kill Python processes on known ports
for %%p in (5001 5002 5003 8080) do (
    for /f "tokens=5" %%a in ('netstat -ano ^| findstr /R "[:.]%%p " ^| findstr "LISTENING"') do (
        if not "%%a"=="" (
            echo Killing process on port %%p ^(PID %%a^)
            taskkill /PID %%a /F >nul 2>&1
        )
    )
)

:: Kill any remaining python.exe
taskkill /IM python.exe /F >nul 2>&1

:: Clean up
if exist "%~dp0running_pids.txt" del "%~dp0running_pids.txt"

echo.
echo ============================================================
echo  All services stopped. Terminals are closed.
echo ============================================================
echo.
timeout /t 2 /nobreak > nul