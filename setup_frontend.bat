@echo off
REM Setup script for Face Recognition System Frontend (Windows)

echo.
echo ================================================
echo Face Recognition System - Frontend Setup
echo ================================================
echo.

REM Check Python version
echo Checking Python version...
python --version

REM Install dependencies
echo.
echo Installing dependencies...
pip install -r requirements_frontend.txt

echo.
echo Setup complete!
echo.
echo To run the application:
echo   python run_frontend.py
echo.
echo Or directly:
echo   python -m frontend.main_window
echo.
pause
