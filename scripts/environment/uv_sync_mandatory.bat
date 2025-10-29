:: Batch script to synchronize only mandatory dependencies (no development dependencies)
@echo off

:: Checks for uv, synchronizes and activates the virtual environment
call uv_check_activate_venv.bat

echo Synchronizing mandatory dependencies
uv sync --native-tls --no-default-groups --no-group "dev"
echo - done
echo.
