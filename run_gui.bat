@echo off
setlocal
call "%~dp0tools\prometheus_env.bat"
if errorlevel 1 exit /b %errorlevel%
cd /d "%~dp0"
"%~dp0build\Release\Prometheus.exe" %*
endlocal
