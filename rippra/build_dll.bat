@echo off
REM build_dll.bat - Build RIPPA shared library (DLL) for Windows
REM Requires MinGW-w64 or MSYS2 with gcc

cd /d "%~dp0"

set CFLAGS=-std=c99 -Wall -Wextra -O2 -DNDEBUG -DBUILD_RIPRA_DLL

if not exist bin mkdir bin

echo Compiling sources...
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

gcc %CFLAGS% -Iinclude -c src\rippra_api.c -o bin\rippra_api.o
if errorlevel 1 exit /b 1

echo Linking rippra.dll...
gcc -shared -o bin\rippra.dll bin\centroid.o bin\io.o bin\la.o bin\recon.o bin\stream.o bin\rippra_api.o -Wl,--out-implib,bin\librippra.dll.a

if errorlevel 1 exit /b 1

echo.
echo === DLL Build successful ===
echo DLL:   bin\rippra.dll
echo Import lib: bin\librippra.dll.a
echo.
echo To use from Python:
echo   import ctypes
echo   dll = ctypes.CDLL(r"bin\rippra.dll")
