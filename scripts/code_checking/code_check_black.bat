:: A script to statically check the AutomationTool code with Black
@echo off

:: Checks for uv, synchronizes and activates the virtual environment
cd "%~dp0..\environment"
call .\uv_sync_all.bat

echo Running black code checking
uv run black .
