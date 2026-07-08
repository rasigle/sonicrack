@echo off
REM Launch the SonicRack Modular Synthesizer
REM
REM This script launches the GUI application

echo ================================================
echo SonicRack - Modular Synthesizer
echo ================================================
echo.

cd "%~dp0"

echo Setting up the environment...
call .\scripts\environment\uv_check_activate_venv.bat

echo Launching the Modular Synthesizer application...
python -m sonicrack.modular_synth_app

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
