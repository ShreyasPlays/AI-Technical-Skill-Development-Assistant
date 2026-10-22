@echo off
cd /d "%~dp0"
title MLflow Tracking UI
echo Starting MLflow Tracking Server...
echo Opening browser at http://127.0.0.1:5000
echo.
call .venv\Scripts\activate.bat
start http://127.0.0.1:5000
mlflow ui --port 5000
pause
