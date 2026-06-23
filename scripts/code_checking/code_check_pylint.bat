:: A script to statically check the repository with Pylint
@echo off
set "script_dir=%~dp0"
for %%I in ("%script_dir%..\..") do set "repo_root=%%~fI"

:: Checks for uv, synchronizes and activates the virtual environment
pushd "%script_dir%..\environment" || exit /b 1
call .\uv_sync_dev.bat
if errorlevel 1 (
    popd
    exit /b 1
)
popd

echo Running pylint
pushd "%repo_root%" || exit /b 1
uv run pylint --rcfile=.pylintrc .\src .\tests .\scripts
set "exit_code=%errorlevel%"
popd
exit /b %exit_code%
