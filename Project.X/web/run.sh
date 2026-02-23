#!/bin/bash

# AQMSS Dashboard Startup Script for Unix/Linux/macOS

echo ""
echo "========================================="
echo "  AQMSS Web Dashboard - Startup Script  "
echo "========================================="
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python 3 is not installed"
    echo "Please install Python 3.8+ from https://python.org"
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "[INFO] Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install/update requirements
echo "[INFO] Installing dependencies..."
pip install -r requirements.txt > /dev/null 2>&1

# Check if data exists
if [ ! -f "../results/market_scores.csv" ]; then
    echo "[WARNING] Market data not found at ../results/market_scores.csv"
    echo "[INFO] The dashboard will show empty until market data is available"
    echo ""
fi

# Run the Flask server
echo ""
echo "[SUCCESS] Starting AQMSS Dashboard..."
echo "[INFO] Dashboard will be available at: http://localhost:5000"
echo "[INFO] Press Ctrl+C to stop the server"
echo ""

python app.py

# Deactivate virtual environment on exit
deactivate
