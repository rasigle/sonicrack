:: A script to statically check and fix the AutomationTool code with ruff
@echo off

:: Checks for uv, synchronizes and activates the virtual environment
cd "%~dp0..\environment"
call .\uv_sync_dev.bat

:: Ruff analyze
echo Running ruff code checking (fix)
uv run ruff check --fix
