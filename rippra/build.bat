@echo off
REM Thin wrapper around CMake build (static library only, no tests)
cd /d "%~dp0.."
cmake -B build -S . -DRIPRA_BUILD_TESTS=OFF -DRIPRA_BUILD_BENCHMARKS=OFF -DRIPRA_BUILD_SHARED=OFF
if errorlevel 1 exit /b 1
cmake --build build --target ripra
