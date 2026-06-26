@echo off
REM build_test_stream.bat - Build streaming test

cd /d "%~dp0"

set CFLAGS=-std=c99 -Wall -Wextra -D_POSIX_SOURCE -O2 -DNDEBUG

echo Building test_stream.exe...
gcc %CFLAGS% -Iinclude tests\test_stream.c -Lbin -lrippra -lm -o bin\test_stream.exe
if errorlevel 1 exit /b 1

echo.
echo === Build successful ===