@echo off
cd /d "%~dp0"
title AI Technical Skill Assistant - Docker Container
echo ======================================================================
echo    AI Technical Skill Development Assistant - Docker Deployment
echo ======================================================================
echo.
echo Building container image and starting services via Docker Compose...
echo.
docker-compose up --build
pause

