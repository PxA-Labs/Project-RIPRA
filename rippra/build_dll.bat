@echo off
REM Thin CMake wrapper: shared library (DLL) with BUILD_RIPRA_DLL
cd /d "%~dp0.."
cmake -B build -S . -DRIPRA_BUILD_SHARED=ON -DRIPRA_BUILD_TESTS=OFF -DRIPRA_BUILD_BENCHMARKS=OFF
if errorlevel 1 exit /b 1
cmake --build build --target ripra
