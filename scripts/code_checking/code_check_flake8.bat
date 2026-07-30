:: A script to statically check code with Flake8
@echo off

:: Checks for uv, synchronizes and activates the virtual environment
cd "%~dp0..\environment"
call .\uv_sync_all.bat

echo Running flake8 code checking
uv run flake8 .\sonicrack .\scripts .\examples .\tests
