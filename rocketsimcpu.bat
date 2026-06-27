@echo off
setlocal

cd /d "%~dp0"

echo Starting PPO train with RocketSim CPU physics backend...
echo Command: build\Release\GigaLearnBot.exe train cpu
echo.

build\Release\GigaLearnBot.exe train cpu

echo.
echo Run finished with exit code %ERRORLEVEL%.
pause
