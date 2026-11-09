@echo off
cd /d "%~dp0"
title AI Technical Skill Assistant - FastAPI Microservice
echo ======================================================================
echo    AI Technical Skill Development Assistant - Phase 2 API Service
echo ======================================================================
echo.
echo Starting FastAPI server at http://127.0.0.1:8000 ...
echo Interactive Swagger Docs: http://127.0.0.1:8000/docs
echo ReDoc Documentation:     http://127.0.0.1:8000/redoc
echo.
python -m uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
pause

