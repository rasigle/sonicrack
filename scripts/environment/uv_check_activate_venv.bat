:: Batch script to automatically detect if uv is installed and available in the PATH.
:: This will be called from other batch scripts to check if uv is installed.
@echo off
set venv_folder=.\.venv

:: Check if 'uv' is installed and available in the PATH
echo Checking if 'uv' is installed...
where uv >nul 2>&1
if errorlevel 1 (
    echo uv is not on PATH
    echo Installing uv...
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    echo uv installed, please re-run this script
    pause
    exit /b 1
) else (
    echo  - uv detected
)
echo.

:: Go to root-folder of the repository and activate the virtual environment
cd "%~dp0..\..

echo Activating virtual environment in %venv_folder%
if not exist %venv_folder% (
    echo "Virtual environment not found. It's recommended to create one using environment batch-script create.bat."
    exit /B 1
)
call %venv_folder%\Scripts\activate
echo.
