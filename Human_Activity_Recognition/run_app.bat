@echo off
echo ========================================
echo  Human Activity Recognition App
echo  Starting FastAPI Server...
echo ========================================
echo.

REM Check if virtual environment exists
if not exist "venv\" (
    echo Creating virtual environment...
    python -m venv venv
    echo.
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Install/upgrade dependencies
echo Installing dependencies...
pip install -r requirements.txt --quiet
echo.

REM Check if dataset exists
if not exist "UCI HAR Dataset\" (
    echo WARNING: Dataset not found!
    echo Please download the UCI HAR Dataset first.
    echo See README.md for instructions.
    echo.
    pause
    exit /b 1
)

REM Start the server
echo Starting server on http://localhost:8000
echo Press Ctrl+C to stop the server
echo.
python backend\main.py

pause
