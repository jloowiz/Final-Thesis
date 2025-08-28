@echo off
REM CARLA Dataset Collection Launcher
REM Windows batch script to collect CARLA training data

echo ================================
echo CARLA Dataset Collection
echo ================================

REM Set default parameters
set OUTPUT=..\..\data\carla_dataset
set HOST=localhost
set PORT=2000
set NUM_IMAGES=1000

REM Parse command line arguments
:parse_args
if "%1"=="--output" (
    set OUTPUT=%2
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
if "%1"=="--num-images" (
    set NUM_IMAGES=%2
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

echo 📋 Collection Configuration:
echo   Output Directory: %OUTPUT%
echo   CARLA Host: %HOST%
echo   CARLA Port: %PORT%
echo   Number of Images: %NUM_IMAGES%
echo.

REM Test CARLA connection
echo 🔄 Testing CARLA connection...
python setup_carla.py --connection-only --host %HOST% --port %PORT%
if errorlevel 1 (
    echo ❌ Cannot connect to CARLA server
    pause
    exit /b 1
)

echo.
echo 🚀 Starting CARLA dataset collection...
echo This will collect %NUM_IMAGES% annotated images
echo Press Ctrl+C to stop early
echo.

REM Run collection
python collect_carla_data.py --output "%OUTPUT%" --host %HOST% --port %PORT% --num-images %NUM_IMAGES%

echo.
echo ✅ Dataset collection complete
echo Check output directory: %OUTPUT%
pause
exit /b 0

:show_help
echo Usage: %0 [options]
echo.
echo Options:
echo   --output PATH       Output directory for dataset
echo   --host HOST         CARLA server host (default: localhost)
echo   --port PORT         CARLA server port (default: 2000)
echo   --num-images N      Number of images to collect (default: 1000)
echo   --help              Show this help message
echo.
pause
exit /b 0
