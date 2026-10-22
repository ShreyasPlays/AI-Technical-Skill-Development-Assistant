@echo off
cd /d "%~dp0"
title AI Technical Skill Development Assistant
echo Starting AI Technical Skill Development Assistant...
echo.
call .venv\Scripts\activate.bat
python predict.py
pause
