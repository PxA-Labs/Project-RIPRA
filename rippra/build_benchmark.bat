@echo off
REM build_benchmark.bat - Build benchmark

cd /d "%~dp0"

set CFLAGS=-std=c99 -Wall -Wextra -D_POSIX_SOURCE -O2 -DNDEBUG

echo Building benchmark.exe...
gcc %CFLAGS% -Iinclude tests\benchmark_openmp.c -Lbin -lrippra -lm -o bin\benchmark_openmp.exe
if errorlevel 1 exit /b 1

echo.
echo === Build successful ===