@echo off
setlocal
cd /d "%~dp0"

call "%~dp0tools\prometheus_env.bat"
if errorlevel 1 exit /b %errorlevel%

set "PYTHON_EXECUTABLE=%PYTHONHOME%\python.exe"
if not exist "%PYTHON_EXECUTABLE%" (
    echo [build] Python executable not found: %PYTHON_EXECUTABLE%
    echo [build] Set PROMETHEUS_PYTHON_HOME to your Python 3.11 folder.
    exit /b 1
)

if not exist "build\CMakeCache.txt" (
if not defined LIBTORCH_DIR (
    if exist "%~dp0libtorch\share\cmake\Torch\TorchConfig.cmake" (
        set "LIBTORCH_DIR=%~dp0libtorch"
    ) else (
        echo [build] Set LIBTORCH_DIR to your LibTorch folder.
        echo [build] Example: set "LIBTORCH_DIR=C:\tools\libtorch"
        exit /b 1
    )
)

    cmake -S . -B build ^
        -DCMAKE_PREFIX_PATH="%LIBTORCH_DIR%" ^
        -DCMAKE_CUDA_FLAGS="-allow-unsupported-compiler" ^
        -DPython_ROOT_DIR="%PYTHONHOME%" ^
        -DPython_EXECUTABLE="%PYTHON_EXECUTABLE%"
    if errorlevel 1 exit /b %errorlevel%
)

cmake --build build --config Release -j
endlocal
