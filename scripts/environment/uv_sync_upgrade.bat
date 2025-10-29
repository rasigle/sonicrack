:: Batch script to synchronize all specified dependencies
@echo off

:: Checks for uv, synchronizes and activates the virtual environment
call uv_check_activate_venv.bat

echo Upgrading all specified dependencies
uv sync --native-tls --upgrade
echo - done
echo.
