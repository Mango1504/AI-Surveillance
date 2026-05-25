@echo off
title AI Surveillance - Startup
echo ===================================================
echo       AI Surveillance System
echo ===================================================
echo.
echo Launching Backend Server...
start "AI Surveillance Backend" cmd /c "Run_Backend.bat"

echo Launching React Dashboard...
cd surveillance-app
start "AI Surveillance Frontend" cmd /k "npm run dev"

echo.
echo Success! 
echo The AI Models are loading (takes ~10 seconds).
echo The dashboard will automatically be available at:
echo http://localhost:3000
echo.
pause
