@echo off
REM build_test_recon.bat - Build test_recon

cd /d "%~dp0"

set CFLAGS=-std=c99 -Wall -Wextra -D_POSIX_SOURCE -O2 -DNDEBUG

echo Building test_recon.exe...
gcc %CFLAGS% -Iinclude tests\test_recon.c -Lbin -lrippra -lm -o bin\test_recon.exe
if errorlevel 1 exit /b 1

echo.
echo === Build successful ===