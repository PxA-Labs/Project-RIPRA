@echo off
REM build_tests_openmp.bat - Build test programs with OpenMP support

cd /d "%~dp0"

set CFLAGS=-std=c99 -Wall -Wextra -D_POSIX_SOURCE -O2 -DNDEBUG -fopenmp
set LDFLAGS=-fopenmp

echo Building tests with OpenMP...

echo Building test_recon.exe...
gcc %CFLAGS% -Iinclude tests\test_recon.c -Lbin -lrippra -lm %LDFLAGS% -o bin\test_recon.exe
if errorlevel 1 exit /b 1

echo Building test_centroid.exe...
gcc %CFLAGS% -Iinclude tests\test_centroid.c -Lbin -lrippra -lm %LDFLAGS% -o bin\test_centroid.exe
if errorlevel 1 exit /b 1

echo Building test_io.exe...
gcc %CFLAGS% -Iinclude tests\test_io.c -Lbin -lrippra -lm %LDFLAGS% -o bin\test_io.exe
if errorlevel 1 exit /b 1

echo Building test_la.exe...
gcc %CFLAGS% -Iinclude tests\test_la.c -Lbin -lrippra -lm %LDFLAGS% -o bin\test_la.exe
if errorlevel 1 exit /b 1

echo.
echo === All tests built successfully ===