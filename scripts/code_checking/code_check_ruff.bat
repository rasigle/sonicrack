:: A script to statically check the AutomationTool code with ruff
@echo off

:: Checks for uv, synchronizes and activates the virtual environment
cd "%~dp0..\environment"
call .\uv_sync_all.bat

:: Ruff analyze
echo Running ruff code checking
uv run ruff check
