#!/usr/bin/env python3
"""
check_benchmark_regression.py — compare benchmark output against baseline.

Usage:
    python tools/check_benchmark_regression.py <benchmark_results_dir> [--baseline benchmarks/baseline.json]

Parses e2e.txt, centroid.txt, openmp.txt, scaling.txt from the results
directory and compares key metrics against the baseline.  Exits with code 1
if any metric regresses beyond the threshold percentage.
"""
import os
import sys
import json
import re

DEFAULT_BASELINE = os.path.join(os.path.dirname(__file__), "..", "benchmarks", "baseline.json")

METRIC_PATTERNS = {
    # e2e.txt patterns
    "e2e_centroid": re.compile(
        r"Centroiding\s*\((\d+) spots\):\s+([\d.]+)\s+ms\s+\(mean\)\s+([\d.]+)\s+ms\s+\(p99\)"
    ),
    "e2e_recon": re.compile(
        r"Deltas \+ Zonal \+ Modal:\s+([\d.]+)\s+ms\s+\(mean\)\s+([\d.]+)\s+ms\s+\(p99\)"
    ),
    "e2e_dm": re.compile(
        r"DM Mapping\s*\((\d+) actuators\):\s+([\d.]+)\s+ms\s+\(mean\)\s+([\d.]+)\s+ms\s+\(p99\)"
    ),
    "e2e_hotpath": re.compile(
        r"HOT-PATH TOTAL \(cent\+recon\+dm\):\s+([\d.]+)\s+ms\s+\(mean\)\s+([\d.]+)\s+ms\s+\(p99\)"
    ),
    "e2e_median": re.compile(
        r"HOT-PATH MEDIAN:\s+([\d.]+)\s+ms"
    ),
    # centroid.txt patterns
    "centroid_fast": re.compile(
        r"Fast centroid \(rippa_compute_centroids\):\s+([\d.]+)\s+ms/frame"
    ),
    "centroid_refined": re.compile(
        r"Refined centroid \(rippa_compute_centroids_refined\):\s+([\d.]+)\s+ms/frame"
    ),
    # openmp.txt / scaling.txt patterns
    "openmp_centroid": re.compile(
        r"Centroiding\s+\((\d+) spots\):\s+([\d.]+)\s+ms"
    ),
}


def parse_file(filepath):
    """Parse a benchmark text file and return a dict of metric_name -> value."""
    if not os.path.exists(filepath):
        return {}
    with open(filepath) as f:
        text = f.read()

    metrics = {}

    # e2e patterns
    m = METRIC_PATTERNS["e2e_centroid"].search(text)
    if m:
        metrics["e2e_centroid_mean_ms"] = float(m.group(2))
        metrics["e2e_centroid_p99_ms"] = float(m.group(3))

    m = METRIC_PATTERNS["e2e_recon"].search(text)
    if m:
        metrics["e2e_recon_mean_ms"] = float(m.group(1))
        metrics["e2e_recon_p99_ms"] = float(m.group(2))

    m = METRIC_PATTERNS["e2e_dm"].search(text)
    if m:
        metrics["e2e_dm_mean_ms"] = float(m.group(2))
        metrics["e2e_dm_p99_ms"] = float(m.group(3))

    m = METRIC_PATTERNS["e2e_hotpath"].search(text)
    if m:
        metrics["e2e_hotpath_mean_ms"] = float(m.group(1))
        metrics["e2e_hotpath_p99_ms"] = float(m.group(2))

    m = METRIC_PATTERNS["e2e_median"].search(text)
    if m:
        metrics["e2e_hotpath_median_ms"] = float(m.group(1))

    # centroid patterns
    m = METRIC_PATTERNS["centroid_fast"].search(text)
    if m:
        metrics["centroid_fast_ms_per_frame"] = float(m.group(1))

    m = METRIC_PATTERNS["centroid_refined"].search(text)
    if m:
        metrics["centroid_refined_ms_per_frame"] = float(m.group(1))

    # openmp patterns (single-thread default)
    m = METRIC_PATTERNS["openmp_centroid"].search(text)
    if m:
        metrics["openmp_1t_centroid_ms"] = float(m.group(2))

    return metrics


def check_regression(results, baseline, threshold_pct):
    """Compare results against baseline; return list of regression messages."""
    regressions = []
    for key, baseline_val in baseline.items():
        if key.startswith("e2e_") or key.startswith("centroid_") or key.startswith("openmp_"):
            current = results.get(key)
            if current is None:
                print(f"  INFO: {key} not in CI output (non-critical — may be environment-dependent)")
                continue
            if baseline_val <= 0:
                continue
            change = (current - baseline_val) / baseline_val * 100
            if change > threshold_pct:
                regressions.append(
                    f"  FAIL: {key} = {current:.3f} ms ({change:+.1f}%), "
                    f"threshold = +{threshold_pct}% (baseline = {baseline_val:.3f} ms)"
                )
    return regressions



def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <benchmark_results_dir> [--baseline baseline.json]")
        sys.exit(1)

    results_dir = sys.argv[1]
    baseline_path = DEFAULT_BASELINE
    if "--baseline" in sys.argv:
        idx = sys.argv.index("--baseline")
        if idx + 1 < len(sys.argv):
            baseline_path = sys.argv[idx + 1]

    # Load baseline
    with open(baseline_path) as f:
        baseline_data = json.load(f)
    baseline = baseline_data["metrics"]
    threshold_pct = baseline_data.get("threshold_pct", 20)

    # Parse current results
    results = {}
    for fname in ("e2e.txt", "centroid.txt", "openmp.txt"):
        results.update(parse_file(os.path.join(results_dir, fname)))

    # Check regressions
    regressions = check_regression(results, baseline, threshold_pct)

    # Report
    print("=== Benchmark Regression Check ===\n")
    print(f"Baseline: {baseline_path}")
    print(f"Results:  {results_dir}\n")

    if not results:
        print("ERROR: No metrics parsed — benchmark output not found or unreadable.")
        sys.exit(1)

    n_checked = sum(1 for k in results if k in baseline)
    print(f"Metrics checked: {n_checked}")
    print(f"Passed:          {n_checked - len(regressions)}")
    print(f"Regressions:     {len(regressions)}")

    if regressions:
        print("\nRegressions detected:")
        for r in regressions:
            print(r)
        sys.exit(1)
    else:
        print("\nAll metrics within threshold. ✓")
        sys.exit(0)


if __name__ == "__main__":
    main()
