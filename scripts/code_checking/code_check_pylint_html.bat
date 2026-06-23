:: A script to statically check the repository with Pylint
@echo off
set "script_dir=%~dp0"
set "json_file=%script_dir%pylint_report.json"
set "html_file=%script_dir%pylint_report.html"
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
if exist "%html_file%" del "%html_file%"
pushd "%repo_root%" || exit /b 1
uv run pylint --rcfile=.pylintrc --output-format=json --output="%json_file%" .\src .\tests .\scripts
if not exist "%json_file%" (
    popd
    echo Pylint did not create JSON report: "%json_file%"
    exit /b 1
)
uv run python -c "import sys; from pylint_json2html import main; sys.exit(main())" -o "%html_file%" "%json_file%"
if errorlevel 1 (
    popd
    exit /b 1
)
popd

if not exist "%html_file%" (
    echo Pylint HTML report was not created: "%html_file%"
    exit /b 1
)
echo Created report: "%html_file%"
start "" "%html_file%"
