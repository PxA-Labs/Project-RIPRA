@echo off
REM build_test_pipeline.bat - Build full pipeline integration test (CI-compatible)

cd /d "%~dp0"

set CFLAGS=-std=c99 -Wall -Wextra -O2 -DNDEBUG

if not exist build mkdir build

echo Compiling...
gcc %CFLAGS% -Iinclude -c src\io.c -o build\io.o
if errorlevel 1 exit /b 1

gcc %CFLAGS% -Iinclude -c src\la.c -o build\la.o
if errorlevel 1 exit /b 1

gcc %CFLAGS% -Iinclude -c src\centroid.c -o build\centroid.o
if errorlevel 1 exit /b 1

gcc %CFLAGS% -Iinclude -c src\recon.c -o build\recon.o
if errorlevel 1 exit /b 1

gcc %CFLAGS% -Iinclude -c src\rippra_api.c -o build\rippra_api.o
if errorlevel 1 exit /b 1

gcc %CFLAGS% -Iinclude -c tests\test_full_pipeline.c -o build\test_full_pipeline.o
if errorlevel 1 exit /b 1

echo Linking...
gcc -o build\test_full_pipeline.exe build\test_full_pipeline.o build\io.o build\la.o build\centroid.o build\recon.o build\rippra_api.o -lm

if errorlevel 1 exit /b 1

echo.
echo === Build successful: build\test_full_pipeline.exe ===
