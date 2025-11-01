@echo off

set venv_folder=..\..\.venv

echo Activating virtual environment
if not exist %venv_folder% (
    echo "Virtual environment not found. It's recommended to create one using environment batch-script create.bat."
    exit /B 1
)
call %venv_folder%\Scripts\activate
echo.

echo Running tests
pytest ..\..\tests --html=html_report\pytest_report.html

if exist html_report\pytest_report.html (
    echo Opening test report
    start html_report\pytest_report.html
) else (
    echo The Coverage report could not be found
)
