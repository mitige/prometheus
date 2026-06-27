@echo off
rem Shared runtime environment for Prometheus launcher scripts.

if not defined PROMETHEUS_PYTHON_HOME (
    if defined PYTHONHOME (
        set "PROMETHEUS_PYTHON_HOME=%PYTHONHOME%"
    ) else (
        for /f "delims=" %%P in ('where python 2^>nul') do (
            for %%I in ("%%P") do set "PROMETHEUS_PYTHON_HOME=%%~dpI"
            goto prometheus_env_python_found
        )
        echo [prometheus] Python was not found. Install Python 3.11 x64 or set PROMETHEUS_PYTHON_HOME.
        exit /b 1
    )
)

:prometheus_env_python_found
set "PYTHONHOME=%PROMETHEUS_PYTHON_HOME%"
set "PYTHONPATH=%PYTHONHOME%\Lib;%PYTHONHOME%\DLLs;%PYTHONHOME%\Lib\site-packages;%~dp0..\build"
set "PATH=%~dp0..\build\Release;%PYTHONHOME%;%PATH%"
exit /b 0
