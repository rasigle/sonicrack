:: A script to statically check the AutomationTool code with MyPy
@echo off

:: Checks for uv, synchronizes and activates the virtual environment
cd "%~dp0..\environment"
call .\uv_sync_dev.bat

echo Running mypy
uv run mypy .\fatlife .\tests
