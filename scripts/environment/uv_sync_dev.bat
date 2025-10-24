:: Batch script to synchronize all specified dependencies
@echo off

:: Checks for uv, synchronizes and activates the virtual environment
call .\uv_check_activate_venv.bat

echo uv Synchronize Development Environment
uv sync --native-tls --group dev
echo - done
echo.


if exist "%venv_folder%\Lib\site-packages\*.pth" (
    echo Removing pth-files in venv-directory
    del %venv_folder%\Lib\site-packages\*.pth
    echo - done
    echo.
)
