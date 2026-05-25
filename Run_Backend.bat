@echo off
title AI Surveillance - Backend
echo ===================================================
echo       AI Surveillance - Backend Server
echo ===================================================
echo.
echo Activating Virtual Environment...
call .venv\Scripts\activate.bat

echo.
echo Starting Backend...
cd surveillance-app\backend
python -u main_proctor.py
pause
