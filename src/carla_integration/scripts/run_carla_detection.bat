@echo off
REM CARLA SSD Detection Launcher
REM Windows batch script to run CARLA SSD detection

echo ================================
echo CARLA SSD Detection System
echo ================================

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Error: Python not found in PATH
    echo Please install Python and add it to PATH
    pause
    exit /b 1
)

REM Set default parameters
set CHECKPOINT=..\..\experiments\synthetic\outputs\models\ssd300_final.pth
set HOST=localhost
set PORT=2000
set CONFIDENCE=0.3

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
if "%1"=="--confidence" (
    set CONFIDENCE=%2
    shift
    shift
    goto parse_args
)
if "%1"=="--help" (
    goto show_help
)
if "%1"=="/?" (
    goto show_help
)
if not "%1"=="" (
    shift
    goto parse_args
)

echo 📋 Configuration:
echo   Checkpoint: %CHECKPOINT%
echo   CARLA Host: %HOST%
echo   CARLA Port: %PORT%
echo   Confidence: %CONFIDENCE%
echo.

REM Check if checkpoint exists
if not exist "%CHECKPOINT%" (
    echo ❌ Error: Checkpoint file not found: %CHECKPOINT%
    echo.
    echo Available checkpoints:
    if exist "..\..\experiments\synthetic\outputs\models\ssd300_final.pth" (
        echo   ✅ ..\..\experiments\synthetic\outputs\models\ssd300_final.pth
    )
    if exist "..\..\experiments\realworld\outputs\models\ssd300_final.pth" (
        echo   ✅ ..\..\experiments\realworld\outputs\models\ssd300_final.pth
    )
    echo.
    echo Use: %0 --checkpoint path\to\checkpoint.pth
    pause
    exit /b 1
)

REM Test CARLA connection
echo 🔄 Testing CARLA connection...
python setup_carla.py --connection-only --host %HOST% --port %PORT%
if errorlevel 1 (
    echo.
    echo ❌ Cannot connect to CARLA server
    echo Please ensure:
    echo   1. CARLA simulator is running
    echo   2. CARLA is accessible at %HOST%:%PORT%
    echo   3. No firewall is blocking the connection
    echo.
    pause
    exit /b 1
)

echo.
echo 🚀 Starting CARLA SSD Detection...
echo Press Ctrl+C to stop
echo.

REM Run detection
python run_carla_detection.py --checkpoint "%CHECKPOINT%" --host %HOST% --port %PORT% --confidence %CONFIDENCE%

echo.
echo ✅ Detection system stopped
pause
exit /b 0

:show_help
echo Usage: %0 [options]
echo.
echo Options:
echo   --checkpoint PATH    Path to SSD model checkpoint
echo   --host HOST         CARLA server host (default: localhost)
echo   --port PORT         CARLA server port (default: 2000)
echo   --confidence CONF   Detection confidence threshold (default: 0.3)
echo   --help              Show this help message
echo.
echo Examples:
echo   %0
echo   %0 --checkpoint model.pth --confidence 0.5
echo   %0 --host 192.168.1.100 --port 2001
echo.
pause
exit /b 0
