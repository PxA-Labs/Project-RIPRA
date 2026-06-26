@echo off
REM build_cuda_test.bat - Build CUDA test program

cd /d "%~dp0"

set CFLAGS=-std=c99 -Wall -Wextra -D_POSIX_SOURCE -O2 -DNDEBUG
set CUFLAGS=-O2

echo Building CUDA test...

if not exist bin mkdir bin

echo Compiling test_cuda.cu...
nvcc %CUFLAGS% -Iinclude -I. tests\test_cuda.c cuda\centroid_kernels.cu cuda\matrix_kernels.cu cuda\dm_kernels.cu src\io.c src\la.c src\centroid.c src\recon.c -o bin\test_cuda.exe -lcudart

if errorlevel 1 (
    echo.
    echo BUILD FAILED - check that CUDA toolkit is installed
    exit /b 1
)

echo.
echo === CUDA test built successfully ===
echo Binary: bin\test_cuda.exe