@echo off
REM Run all tests for the chess project

echo Running Chess Game Tests...
echo ============================

cd backend

REM Check if pytest is available
python -c "import pytest" 2>nul
if errorlevel 1 (
    echo Installing test dependencies...
    pip install -r ../requirements.txt
)

REM Run tests
echo.
echo Running tests with pytest...
python -m pytest tests/ -v

REM Check exit code
if %ERRORLEVEL% == 0 (
    echo.
    echo ✅ All tests passed!
) else (
    echo.
    echo ❌ Some tests failed!
    exit /b 1
)
