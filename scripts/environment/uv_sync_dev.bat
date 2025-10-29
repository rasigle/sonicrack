:: Batch script to synchronize all specified dependencies
@echo off

:: Checks for uv, synchronizes and activates the virtual environment
call .\uv_check_activate_venv.bat

echo uv Synchronize Development Environment
uv sync --native-tls --group dev --group examples
echo - done
echo.
