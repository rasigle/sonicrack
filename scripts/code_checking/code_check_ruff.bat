:: A script to statically check the AutomationTool code with ruff
@echo off
set log_file="%~dp0ruff.log"

:: Checks for uv, synchronizes and activates the virtual environment
cd "%~dp0..\environment"
call .\uv_sync_dev.bat

:: Ruff analyze
echo Running ruff code checking
if exist %log_file% del %log_file%
uv run ruff check > %log_file%
type %log_file%
echo Created log-file: %log_file%
