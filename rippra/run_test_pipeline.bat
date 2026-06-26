@echo off
cd /d "%~dp0.."
rippra\bin\test_full_pipeline.exe .
if errorlevel 1 (
    echo.
    echo Some tests FAILED.
    exit /b 1
) else (
    echo.
    echo All tests PASSED.
    exit /b 0
)
