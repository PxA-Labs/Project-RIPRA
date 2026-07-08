#!/usr/bin/env python3
"""
reproduce_all.py - Automates calibration, lightweight dataset generation,
model training, and ONNX validation checks to ensure full reproducibility of the ML pipeline.
"""
import os
import subprocess
import sys

def run_command(cmd, shell=False):
    print(f"\nExecuting: {cmd}")
    res = subprocess.run(cmd, shell=shell, capture_output=True, encoding='utf-8')
    if res.returncode != 0:
        print(f"ERROR: command failed with code {res.returncode}")
        print(f"STDOUT:\n{res.stdout}")
        print(f"STDERR:\n{res.stderr}")
        sys.exit(res.returncode)
    print("SUCCESS")
    return res.stdout

def main():
    try:
        import sys
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        pass
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(base_dir)
    print(f"Current working directory: {os.getcwd()}")

    # 1. Run C calibration/centroid test to generate results/reference_centroids_c.csv
    import platform
    is_windows = (platform.system() == 'Windows')
    
    lib_path = os.path.join("bin", "librippra.a") if is_windows else os.path.join("build", "librippra.a")
    centroid_exe = os.path.join("bin", "test_centroid.exe") if is_windows else os.path.join("bin", "test_centroid")
    
    os.makedirs("bin", exist_ok=True)
    os.makedirs("build", exist_ok=True)

    print("--- Step 1: Ensure C library and calibration tools are built ---")
    if not os.path.exists(lib_path):
        print(f"  Building static C library: {lib_path}")
        src_files = ["io.c", "la.c", "centroid.c", "recon.c", "rippra_api.c", "simd.c"]
        for f in src_files:
            obj = f.replace(".c", ".o")
            run_command(f"gcc -std=c99 -Wall -Wextra -O2 -DNDEBUG -Iinclude -c src/{f} -o build/{obj}", shell=True)
        run_command(f"gcc -std=c99 -Wall -Wextra -O2 -DNDEBUG -Iinclude -mavx2 -mfma -c src/simd_avx2.c -o build/simd_avx2.o", shell=True)
        objs_str = " ".join([f"build/{f.replace('.c', '.o')}" for f in src_files]) + " build/simd_avx2.o"
        run_command(f"ar rcs {lib_path} {objs_str}", shell=True)
        
    if not os.path.exists(centroid_exe):
        print(f"  Compiling calibration tool: {centroid_exe}")
        run_command(f"gcc -std=c99 -Wall -Wextra -D_POSIX_SOURCE -O2 -DNDEBUG -Iinclude tests/test_centroid.c {lib_path} -lm -o {centroid_exe}", shell=True)
        
    os.makedirs("results", exist_ok=True)
    run_command([centroid_exe])

    # 2. Generate a lightweight ML dataset (500 samples)
    print("\n--- Step 2: Generate lightweight Kolmogorov dataset (500 samples) ---")
    run_command([
        sys.executable, "tools/generate_dataset.py",
        "--samples", "500",
        "--out", "data_ai/dataset.npz",
        "--noise", "0.1",
        "--seed", "42"
    ])

    # 3. Train a lightweight MLP reconstructor (3 epochs) to create PyTorch checkpoints
    print("\n--- Step 3: Train MLP model for 3 epochs ---")
    run_command([
        sys.executable, "ml/train.py",
        "--model", "mlp",
        "--epochs", "3",
        "--batch_size", "32",
        "--lr", "1e-3",
        "--dataset", "data_ai/dataset.npz",
        "--out_dir", "ml_checkpoints/local"
    ])

    # 4. Train CNN model (3 epochs)
    print("\n--- Step 4: Train CNN model for 3 epochs ---")
    run_command([
        sys.executable, "ml/train.py",
        "--model", "cnn",
        "--epochs", "3",
        "--batch_size", "32",
        "--lr", "1e-3",
        "--dataset", "data_ai/dataset.npz",
        "--out_dir", "ml_checkpoints/local"
    ])

    # 5. Export to ONNX
    print("\n--- Step 5: Export trained models to ONNX ---")
    run_command([
        sys.executable, "ml/export_onnx.py",
        "--output_dir", "onnx_models"
    ])

    # 6. Evaluate model accuracy (MSE + correlation vs classical)
    print("\n--- Step 6: Evaluate model accuracy ---")
    run_command([sys.executable, "ml/evaluate_inference.py"])

    # 7. Validate ONNX models
    print("\n--- Step 7: Validate ONNX models ---")
    run_command([sys.executable, "ml/test_onnx_models.py"])

    # 8. Run Predictive AO evaluation
    print("\n--- Step 8: Run predictive AO simulation ---")
    run_command([sys.executable, "ml/predictive_ao.py"])

    # 9. Benchmark ONNX model inference latency
    print("\n--- Step 9: Benchmark ONNX inference latency ---")
    run_command([sys.executable, "ml/benchmark_onnx_latency.py"])

    print("\n=======================================================")
    print("=== All reproducibility steps executed successfully ===")
    print("=======================================================")

if __name__ == "__main__":
    main()
