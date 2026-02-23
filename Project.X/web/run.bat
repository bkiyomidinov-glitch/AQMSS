@echo off
REM AQMSS Dashboard Startup Script for Windows

echo.
echo =========================================
echo   AQMSS Web Dashboard - Startup Script
echo =========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://python.org
    pause
    exit /b 1
)

REM Check if virtual environment exists
if not exist venv (
    echo [INFO] Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Install/update requirements
echo [INFO] Installing dependencies...
pip install -r requirements.txt > nul 2>&1

REM Check if data exists
if not exist ..\results\market_scores.csv (
    echo [WARNING] Market data not found at ..\results\market_scores.csv
    echo [INFO] The dashboard will show empty until market data is available
    echo.
)

REM Run the Flask server
echo.
echo [SUCCESS] Starting AQMSS Dashboard...
echo [INFO] Dashboard will be available at: http://localhost:5000
echo [INFO] Press Ctrl+C to stop the server
echo.

python app.py

REM Deactivate virtual environment on exit
call venv\Scripts\deactivate.bat

pause
