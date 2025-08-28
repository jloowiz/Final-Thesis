@echo off
REM CARLA Performance Benchmark Launcher
REM Windows batch script to run CARLA SSD benchmarks

echo ================================
echo CARLA SSD Benchmark System
echo ================================

REM Set default parameters
set CHECKPOINT=..\..\experiments\synthetic\outputs\models\ssd300_final.pth
set HOST=localhost
set PORT=2000
set DURATION=30

REM Parse command line arguments
:parse_args
if "%1"=="--checkpoint" (
    set CHECKPOINT=%2
    shift
    shift
    goto parse_args
)
if "%1"=="--host" (
    set HOST=%2
    shift
    shift
    goto parse_args
)
if "%1"=="--port" (
    set PORT=%2
    shift
    shift
    goto parse_args
)
if "%1"=="--duration" (
    set DURATION=%2
    shift
    shift
    goto parse_args
)
if "%1"=="--help" (
    goto show_help
)
if not "%1"=="" (
    shift
    goto parse_args
)

echo 📋 Benchmark Configuration:
echo   Checkpoint: %CHECKPOINT%
echo   CARLA Host: %HOST%
echo   CARLA Port: %PORT%
echo   Duration per scenario: %DURATION% seconds
echo.

REM Check if checkpoint exists
if not exist "%CHECKPOINT%" (
    echo ❌ Error: Checkpoint file not found: %CHECKPOINT%
    pause
    exit /b 1
)

REM Test CARLA connection
echo 🔄 Testing CARLA connection...
python setup_carla.py --connection-only --host %HOST% --port %PORT%
if errorlevel 1 (
    echo ❌ Cannot connect to CARLA server
    pause
    exit /b 1
)

echo.
echo 🚀 Starting CARLA SSD Benchmark...
echo This will test performance across multiple scenarios
echo.

REM Run benchmark
python benchmark_carla.py --checkpoint "%CHECKPOINT%" --host %HOST% --port %PORT% --duration %DURATION%

echo.
echo ✅ Benchmark complete
pause
exit /b 0

:show_help
echo Usage: %0 [options]
echo.
echo Options:
echo   --checkpoint PATH    Path to SSD model checkpoint
echo   --host HOST         CARLA server host (default: localhost)
echo   --port PORT         CARLA server port (default: 2000)
echo   --duration SECONDS  Duration per scenario (default: 30)
echo   --help              Show this help message
echo.
pause
exit /b 0
