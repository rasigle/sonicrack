@echo off

set venv_folder=..\..\.venv

echo Activating virtual environment
if not exist %venv_folder% (
    echo "Virtual environment not found. It's recommended to create one using environment batch-script create.bat."
    exit /B 1
)
call %venv_folder%\Scripts\activate
echo.

echo Running tests and creating coverage report
pytest --cov=..\..\fatlife --cov-report=html ..\..\tests

if exist htmlcov\index.html (
    echo Opening coverage report
    start htmlcov\index.html
) else (
    echo The Coverage report could not be found
)
