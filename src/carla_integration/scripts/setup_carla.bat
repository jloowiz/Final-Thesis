@echo off
REM CARLA Setup and Testing Launcher
REM Windows batch script for CARLA environment setup

echo ================================
echo CARLA Setup and Testing
echo ================================

REM Set default parameters
set HOST=localhost
set PORT=2000
set ACTION=default

REM Parse command line arguments
:parse_args
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
if "%1"=="--setup-only" (
    set ACTION=setup
    shift
    goto parse_args
)
if "%1"=="--connection-only" (
    set ACTION=connection
    shift
    goto parse_args
)
if "%1"=="--full-test" (
    set ACTION=full-test
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

echo 📋 Configuration:
echo   CARLA Host: %HOST%
echo   CARLA Port: %PORT%
echo   Test Mode: %ACTION%
echo.

if "%ACTION%"=="setup" (
    echo 🔧 Running setup instructions only...
    python setup_carla.py --setup-only
) else if "%ACTION%"=="connection" (
    echo 🔄 Testing connection only...
    python setup_carla.py --connection-only --host %HOST% --port %PORT%
) else if "%ACTION%"=="full-test" (
    echo 🧪 Running full test suite...
    python setup_carla.py --full-test --host %HOST% --port %PORT%
) else (
    echo 🔧 Running default setup and connection test...
    python setup_carla.py --host %HOST% --port %PORT%
)

echo.
pause
exit /b 0

:show_help
echo Usage: %0 [options]
echo.
echo Options:
echo   --host HOST         CARLA server host (default: localhost)
echo   --port PORT         CARLA server port (default: 2000)
echo   --setup-only        Only show setup instructions
echo   --connection-only   Only test connection
echo   --full-test         Run full test suite
echo   --help              Show this help message
echo.
echo Examples:
echo   %0                      # Default setup and basic test
echo   %0 --connection-only    # Quick connection test
echo   %0 --full-test          # Complete test suite
echo   %0 --setup-only         # Just show setup instructions
echo.
pause
exit /b 0
