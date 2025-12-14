#!/bin/bash

echo "========================================"
echo " Human Activity Recognition App"
echo " Starting FastAPI Server..."
echo "========================================"
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo ""
fi

# Activate virtual environment
source venv/bin/activate

# Install/upgrade dependencies
echo "Installing dependencies..."
pip install -r requirements.txt --quiet
echo ""

# Check if dataset exists
if [ ! -d "UCI HAR Dataset" ]; then
    echo "WARNING: Dataset not found!"
    echo "Please download the UCI HAR Dataset first."
    echo "See README.md for instructions."
    echo ""
    read -p "Press enter to continue anyway or Ctrl+C to exit..."
fi

# Start the server
echo "Starting server on http://localhost:8000"
echo "Press Ctrl+C to stop the server"
echo ""
python backend/main.py
