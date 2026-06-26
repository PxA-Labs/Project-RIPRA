@echo off
REM build_openmp.bat - Build RIPPA with OpenMP support

cd /d "%~dp0"

set CFLAGS=-std=c99 -Wall -Wextra -D_POSIX_SOURCE -O2 -DNDEBUG -fopenmp
set LDFLAGS=-fopenmp

echo Building with OpenMP...
echo CFLAGS: %CFLAGS%

if not exist bin mkdir bin

echo Compiling centroid.c...
gcc %CFLAGS% -Iinclude -c src\centroid.c -o bin\centroid.o
if errorlevel 1 exit /b 1

echo Compiling io.c...
gcc %CFLAGS% -Iinclude -c src\io.c -o bin\io.o
if errorlevel 1 exit /b 1

echo Compiling la.c...
gcc %CFLAGS% -Iinclude -c src\la.c -o bin\la.o
if errorlevel 1 exit /b 1

echo Compiling recon.c...
gcc %CFLAGS% -Iinclude -c src\recon.c -o bin\recon.o
if errorlevel 1 exit /b 1

echo Linking librippra.a...
ar rcs bin\librippra.a bin\centroid.o bin\io.o bin\la.o bin\recon.o
ranlib bin\librippra.a 2>nul

echo.
echo === Build successful ===
echo Library: bin\librippra.a
echo.
echo To build a test program, link with:
echo gcc %CFLAGS% -Iinclude your_main.c -Lbin -lrippra -lm %LDFLAGS% -o bin\your_program