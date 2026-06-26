@echo off
cd /d "%~dp0"

echo Testing OpenMP compilation...

gcc -std=c99 -Wall -Wextra -D_POSIX_SOURCE -O2 -DNDEBUG -fopenmp -Iinclude -c src\centroid.c -o bin\centroid_omp.o
if errorlevel 1 (
    echo centroid.c: FAILED
) else (
    echo centroid.c: OK
)

gcc -std=c99 -Wall -Wextra -D_POSIX_SOURCE -O2 -DNDEBUG -fopenmp -Iinclude -c src\la.c -o bin\la_omp.o
if errorlevel 1 (
    echo la.c: FAILED
) else (
    echo la.c: OK
)

gcc -std=c99 -Wall -Wextra -D_POSIX_SOURCE -O2 -DNDEBUG -fopenmp -Iinclude -c src\recon.c -o bin\recon_omp.o
if errorlevel 1 (
    echo recon.c: FAILED
) else (
    echo recon.c: OK
)

echo.
echo All files compiled with -fopenmp successfully!