:: Batch script to synchronize all specified dependencies
@echo off

:: Checks for uv, synchronizes and activates the virtual environment
call .\uv_check_activate_venv.bat

echo uv Synchronize Environment (all groups)
uv sync --extra full --all-groups
echo - done
echo.
