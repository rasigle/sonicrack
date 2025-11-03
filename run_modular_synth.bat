@echo off
REM Launch the AudioPlayground Modular Synthesizer
REM
REM This script launches the GUI application

echo ================================================
echo AudioPlayground - Modular Synthesizer
echo ================================================
echo.

cd /d "%~dp0"

echo Setting up the environment...
call .\scripts\environment\uv_check_activate_venv.bat

echo Launching the Modular Synthesizer application...
cd src
python modular_synth_app.py

if errorlevel 1 (
    echo.
    echo ================================================
    echo Error: Failed to launch the application
    echo.
    echo Make sure you have installed all dependencies:
    echo   uv sync
    echo ================================================
    pause
)

