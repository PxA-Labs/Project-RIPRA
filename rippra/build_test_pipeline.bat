@echo off
REM build_test_pipeline.bat - Build full pipeline integration test

cd /d "%~dp0"

set CFLAGS=-std=c99 -Wall -Wextra -O2 -DNDEBUG

if not exist bin mkdir bin

echo Compiling...
gcc %CFLAGS% -Iinclude -c src\centroid.c -o bin\centroid.o
if errorlevel 1 exit /b 1

gcc %CFLAGS% -Iinclude -c src\io.c -o bin\io.o
if errorlevel 1 exit /b 1

gcc %CFLAGS% -Iinclude -c src\la.c -o bin\la.o
if errorlevel 1 exit /b 1

gcc %CFLAGS% -Iinclude -c src\recon.c -o bin\recon.o
if errorlevel 1 exit /b 1

gcc %CFLAGS% -Iinclude -c src\stream.c -o bin\stream.o
if errorlevel 1 exit /b 1

gcc %CFLAGS% -Iinclude -c tests\test_full_pipeline.c -o bin\test_full_pipeline.o
if errorlevel 1 exit /b 1

echo Linking...
gcc -o bin\test_full_pipeline.exe bin\test_full_pipeline.o bin\centroid.o bin\io.o bin\la.o bin\recon.o bin\stream.o -lm

if errorlevel 1 exit /b 1

echo.
echo === Build successful: bin\test_full_pipeline.exe ===
