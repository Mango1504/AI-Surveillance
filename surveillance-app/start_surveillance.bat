@echo off
echo Starting AI Surveillance System...
set PYTHONIOENCODING=utf-8

echo Starting Backend API...
start cmd /k "cd backend && py -u main_proctor.py"

echo Starting React Frontend...
start cmd /k "npm.cmd run dev"

echo System is launching! The backend will take about 15 seconds to initialize the camera and YOLO models.
echo The dashboard will be available at http://localhost:3000
