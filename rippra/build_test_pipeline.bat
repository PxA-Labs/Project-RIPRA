@echo off
REM Thin CMake wrapper: build and optionally run all tests
cd /d "%~dp0.."
cmake -B build -S . -DRIPRA_BUILD_TESTS=ON -DRIPRA_BUILD_BENCHMARKS=OFF -DRIPRA_BUILD_SHARED=OFF
if errorlevel 1 exit /b 1
cmake --build build
if errorlevel 1 exit /b 1
echo.
echo === Build successful ===
echo Run tests with: ctest --test-dir build --output-on-failure
