"""Benchmark ONNX model inference latency under realistic deployment constraints.

Loads each ONNX model from onnx_models/, runs warmup + 1000 timed iterations
on CPU (and GPU if CUDA Execution Provider is available), reports mean,
median, p99, p999 latency, hardware spec, and states whether ML-in-the-loop
meets the 10 ms real-time budget.
"""
import os
import sys
import time
import csv
import platform

try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except AttributeError:
    pass

try:
    import numpy as np
    import onnxruntime as ort
except ImportError:
    print("SKIP: onnxruntime not installed")
    sys.exit(0)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ONNX_DIR = os.path.join(BASE, "onnx_models")
RESULTS_DIR = os.path.join(BASE, "results")
N_WARMUP = 100
N_ITER = 1000

def get_hardware_info():
    info = {"cpu": platform.processor() or "unknown"}
    try:
        import psutil
        info["cpu_cores"] = psutil.cpu_count(logical=True)
        info["cpu_freq"] = psutil.cpu_freq().max if psutil.cpu_freq() else "unknown"
    except ImportError:
        pass
    try:
        import subprocess
        result = subprocess.run(
            ["wmic", "cpu", "get", "name"],
            capture_output=True, text=True, timeout=5
        )
        lines = [l.strip() for l in result.stdout.splitlines() if l.strip() and "Name" not in l]
        if lines:
            info["cpu_model"] = lines[0]
    except Exception:
        pass
    try:
        import torch
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["cuda_version"] = torch.version.cuda
    except ImportError:
        providers = ort.get_available_providers()
        cuda_provs = [p for p in providers if "CUDA" in p]
        if cuda_provs:
            info["gpu"] = "CUDA EP available (ort)"
    return info

def get_ort_session(onnx_path, use_gpu=False):
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if use_gpu else ["CPUExecutionProvider"]
    available = ort.get_available_providers()
    if use_gpu and "CUDAExecutionProvider" not in available:
        return None
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    try:
        return ort.InferenceSession(onnx_path, opts, providers=providers)
    except Exception:
        return None

def benchmark_model(onnx_path, model_name, use_gpu=False):
    sess = get_ort_session(onnx_path, use_gpu=use_gpu)
    if sess is None:
        return None

    inp = sess.get_inputs()[0]
    inp_shape = list(inp.shape)
    iname = inp.name

    # Build dummy input matching expected shape (batch=1)
    dummy_shape = [1 if not isinstance(d, int) else d for d in inp_shape]
    dummy = np.random.randn(*dummy_shape).astype(np.float32)

    label = f"{model_name} (GPU)" if use_gpu else f"{model_name} (CPU)"

    # Warmup
    for _ in range(N_WARMUP):
        sess.run(None, {iname: dummy})

    # Timed iterations
    times = []
    for _ in range(N_ITER):
        t0 = time.perf_counter()
        sess.run(None, {iname: dummy})
        times.append(time.perf_counter() - t0)

    times_ms = np.array(times) * 1000.0
    stats = {
        "model": label,
        "provider": "CUDA" if use_gpu else "CPU",
        "mean_ms": float(np.mean(times_ms)),
        "median_ms": float(np.median(times_ms)),
        "std_ms": float(np.std(times_ms)),
        "min_ms": float(np.min(times_ms)),
        "max_ms": float(np.max(times_ms)),
        "p99_ms": float(np.percentile(times_ms, 99)),
        "p999_ms": float(np.percentile(times_ms, 99.9)),
        "n_iter": N_ITER,
        "input_shape": str(inp_shape),
    }
    return stats

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 60)
    print("  ONNX Inference Latency Benchmark")
    print("=" * 60)

    hw = get_hardware_info()
    print(f"\n  Hardware:")
    print(f"    CPU: {hw.get('cpu_model', hw['cpu'])}")
    print(f"    Cores: {hw.get('cpu_cores', 'N/A')}")
    if "gpu" in hw:
        print(f"    GPU: {hw['gpu']}")
    if "cuda_version" in hw:
        print(f"    CUDA: {hw['cuda_version']}")

    # Scan ONNX dir for models
    if not os.path.isdir(ONNX_DIR):
        print(f"\nSKIP: {ONNX_DIR} not found — run export_onnx.py first")
        sys.exit(0)

    model_files = sorted(f for f in os.listdir(ONNX_DIR) if f.endswith(".onnx"))
    if not model_files:
        print(f"\nNo .onnx files found in {ONNX_DIR}")
        sys.exit(0)

    # Check for GPU availability
    providers = ort.get_available_providers()
    has_gpu = "CUDAExecutionProvider" in providers
    print(f"\n  ONNX Runtime providers: {providers}")
    print(f"  GPU available: {'YES' if has_gpu else 'NO'}")

    all_stats = []
    print(f"\n  {'Model':<25s} {'Provider':<8s} {'Mean(ms)':<10s} {'Median(ms)':<11s} {'p99(ms)':<9s} {'p999(ms)':<10s}")
    print(f"  {'-'*25} {'-'*8} {'-'*10} {'-'*11} {'-'*9} {'-'*10}")

    classic_hotpath_us = 761.0  # from C e2e benchmark

    for fname in model_files:
        path = os.path.join(ONNX_DIR, fname)
        model_name = fname.replace(".onnx", "")
        for use_gpu in [False, True]:
            stats = benchmark_model(path, model_name, use_gpu=use_gpu)
            if stats is None:
                continue
            all_stats.append(stats)
            print(f"  {stats['model']:<25s} {stats['provider']:<8s} {stats['mean_ms']:<10.4f} {stats['median_ms']:<11.4f} {stats['p99_ms']:<9.4f} {stats['p999_ms']:<10.4f}")

    if not all_stats:
        print("\n  No models could be benchmarked.")
        sys.exit(0)

    # Save CSV
    csv_path = os.path.join(RESULTS_DIR, "onnx_latency_benchmark.csv")
    fieldnames = ["model", "provider", "mean_ms", "median_ms", "std_ms",
                  "min_ms", "max_ms", "p99_ms", "p999_ms", "n_iter", "input_shape"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(all_stats)
    print(f"\n  Results saved: results/onnx_latency_benchmark.csv")

    # Budget compliance
    print(f"\n  {'='*60}")
    print(f"  Real-Time Budget Analysis (10 ms target)")
    print(f"  {'='*60}")
    print(f"  C classical hot-path: {classic_hotpath_us/1000:.4f} ms")

    for s in all_stats:
        within_budget = s["mean_ms"] < 10.0
        msg = "PASS" if within_budget else "FAIL"
        extra = ""
        if not within_budget:
            extra = f" — exceeds 10 ms budget by {s['mean_ms'] - 10.0:.2f} ms"
        else:
            # Show combined C + ML overhead
            combined = classic_hotpath_us / 1000 + s["mean_ms"]
            combined_ok = combined < 10.0
            cmsg = "PASS" if combined_ok else "FAIL"
            extra = f" (C+ML combined {combined:.3f} ms → {cmsg})"
        print(f"  {s['model']:<25s} ({s['provider']}): mean={s['mean_ms']:.4f} ms → [{msg}]{extra}")

    print(f"  {'='*60}")

if __name__ == "__main__":
    main()
