@echo off
REM build.bat - Build RIPPA C pipeline (release)

cd /d "%~dp0"

set CFLAGS=-std=c99 -Wall -Wextra -D_POSIX_SOURCE -O2 -DNDEBUG

echo Building release...
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

echo Compiling stream.c...
gcc %CFLAGS% -Iinclude -c src\stream.c -o bin\stream.o
if errorlevel 1 exit /b 1

echo Linking librippra.a...
ar rcs bin\librippra.a bin\centroid.o bin\io.o bin\la.o bin\recon.o bin\stream.o
ranlib bin\librippra.a 2>nul

echo.
echo === Build successful ===
echo Library: bin\librippra.a