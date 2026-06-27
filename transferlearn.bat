@echo off
setlocal
cd /d "%~dp0"

call "%~dp0tools\prometheus_env.bat"
if errorlevel 1 exit /b %errorlevel%
set "CUDA_VISIBLE_DEVICES=0"
set "CUDA_MODULE_LOADING=LAZY"
set "NVIDIA_TF32_OVERRIDE=1"
set "OMP_NUM_THREADS=%NUMBER_OF_PROCESSORS%"
set "MKL_NUM_THREADS=%NUMBER_OF_PROCESSORS%"

if not exist "build\Release\GigaLearnBot.exe" (
    echo [transferlearn] GigaLearnBot.exe not found. Building Release...
    call "%~dp0build.bat"
    if errorlevel 1 exit /b %errorlevel%
)

if not exist "checkpoints\28188091392\POLICY.lt" (
    echo [transferlearn] Missing teacher checkpoint: checkpoints\28188091392
    exit /b 1
)

if not exist "checkpoints_transfer" (
    mkdir "checkpoints_transfer"
)

echo [transferlearn] Teacher: checkpoints\28188091392
echo [transferlearn] Student checkpoints: checkpoints_transfer

if /I "%~1"=="once" goto run_once
if /I "%~1"=="loop" goto run_loop_mode
if /I "%~1"=="raw-full" goto run_direct
if /I "%~1"=="direct" goto run_direct

goto run_direct

:run_loop_mode
echo [transferlearn] Starting restart-loop transfer learning.
echo [transferlearn] This legacy mode runs one transfer iteration per process and resumes from the newest student checkpoint.
echo [transferlearn] Press Ctrl+C to stop between iterations.
set /a LOOP_INDEX=0
:run_loop
set /a LOOP_INDEX+=1
echo.
echo [transferlearn] ===== Iteration %LOOP_INDEX% =====
"%~dp0build\Release\GigaLearnBot.exe" transfer-once
set "EXIT_CODE=%ERRORLEVEL%"
if not "%EXIT_CODE%"=="0" goto done

if not "%TRANSFERLEARN_MAX_LOOPS%"=="" (
    if %LOOP_INDEX% GEQ %TRANSFERLEARN_MAX_LOOPS% goto done
)

goto run_loop

:run_once
echo [transferlearn] Starting one transfer learning iteration.
"%~dp0build\Release\GigaLearnBot.exe" transfer-once
set "EXIT_CODE=%ERRORLEVEL%"
goto done

:run_direct
echo [transferlearn] Starting continuous in-process transfer learning.
echo [transferlearn] This keeps RocketSim/CUDA/model state initialized between iterations.
"%~dp0build\Release\GigaLearnBot.exe" transfer
set "EXIT_CODE=%ERRORLEVEL%"

:done
echo [transferlearn] Exit code: %EXIT_CODE%
exit /b %EXIT_CODE%
