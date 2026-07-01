# Project RIPRA — Expert Panel Technical Audit, Gap Analysis & GitHub Issue Backlog

**Repository:** `github.com/PxA-Labs/Project-RIPRA`
**Target:** ISRO Bharatiya Antariksh Hackathon 2026 — Problem Statement 9 (Wavefront Reconstruction, Turbulence Characterization, DM Actuator Mapping)
**Audit basis:** Full clone and source inspection (not GitHub-page metadata) — `rippra/src/*.c`, `rippra/include/*.h`, `rippra/cuda/*.cu`, `rippra/ml/*.py`, `rippra/tests/*.c`, `docs/*`, `CMakeLists.txt`, `.github/workflows/ci.yml`, `config/default.yaml`.
**Audit date:** July 1, 2026

> **Note on a prior audit in this repo:** `docs/RIPRA_Audit_Report_BAH2026.md` (dated June 28, 2026) scored the project 4.8/10 and "DO NOT SUBMIT," based on GitHub-page metadata rather than the actual source. That audit is now materially stale — `LICENSE`, `config/`, CUDA kernels, an ML pipeline, and a full C test suite all exist in the current tree. This audit supersedes it and is grounded in the code as it exists today. Treat the June 28 report as historical.

---

## 1. Executive Summary

Project RIPRA is **substantially further along than a skeleton**. The C core (`centroid.c`, `la.c`, `recon.c`, `io.c`, `stream.c`, `rippra_api.c`) implements real, working versions of every core ISRO deliverable: TCoG centroiding with two-pass refinement, Fried-geometry zonal reconstruction, modal (Zernike) reconstruction, r₀ and τ₀ estimation, DM actuator mapping with nearest-neighbor/diagonal coupling, and a closed-loop control iteration. A vendored one-sided Jacobi SVD and Doolittle LU avoid an external LAPACK dependency. CUDA kernels exist for centroiding, DM operations, and matrix math. An ML side-pipeline (MLP/CNN/LSTM reconstructors, ONNX export, ablation studies, noise-robustness sweeps) is present and reasonably mature. CI builds and runs integration tests on Linux and Windows.

This audit does **not** repeat the "critical blockers" from the stale June 28 report, most of which no longer apply. Instead it focuses on what a hackathon judge, an IEEE/SPIE reviewer, and a production AO engineer would still flag:

1. **Validation, not just implementation.** r₀, τ₀, Strehl ratio, and reconstruction accuracy are computed but not validated against independent ground truth (known-turbulence synthetic cases with analytically expected r₀/τ₀, or literature benchmark numbers). "The code runs and produces a number" ≠ "the number is correct."
2. **Architectural fragility.** `rippra_api.c` duplicates struct definitions from `recon.h` instead of including it, because the two headers currently declare incompatible types under the same names. This is a landmine — any change to `rippra_zonal_mesh` in `recon.h` silently desynchronizes from the copy in `rippra_api.c` with no compiler error.
3. **Untested failure paths.** `malloc` results are inconsistently checked; `fread` return values are discarded in the BMP reader; NaN centroids are zeroed rather than flagged, silently injecting a false "measured phase" into reconstruction when a spot is lost.
4. **CI covers ~40% of the surface.** CUDA kernels, GPU benchmarks, closed-loop control, predictive AO, and PyTorch training paths are not exercised by CI. `test_cuda.c` and the `cuda/` kernels exist but never run on a GitHub-hosted runner (no GPU runner configured).
5. **Real dataset compatibility is asserted, not demonstrated.** The BMP/raw readers and `config/default.yaml` describe an assumed sensor/MLA/DM geometry; there's no evidence in-repo that this has been run against the actual ISRO-released dataset (which, per the problem statement, hasn't been released before submission in prior cycles either, but the assumptions should be documented and defended, not implicit).
6. **GitHub project hygiene is minimal.** One CI workflow, no issue/PR templates, no labels config, no CodeQL/dependency scanning, no branch protection evidence, no project board, no milestones.

**Verdict:** 🟡 **Competitive with hardening, not yet "submission-polished."** The core algorithmic bar for PS9 appears cleared. The remaining work is verification, robustness, CI breadth, and documentation/presentation polish — exactly the kind of work that separates a top-10 hackathon submission from a top-40 one with equivalent algorithms.

**Estimated completion:** 65–75% of a fully hardened, validated, judge-ready submission.
**Overall score (this audit):** 6.8 / 10 — see full scorecard in §18.

---

## 2. Repository Audit

### Structure (as cloned)

```
Project-RIPRA/
├── Audit.tex, LICENSE, SECURITY.md, CONTRIBUTING.md, CHANGELOG.md, AGENTS.md
├── CMakeLists.txt, Dockerfile
├── config/default.yaml
├── docs/            (11 markdown files + docs/paper/)
├── notebook/         (5 Jupyter notebooks, 1.9 MB total)
├── rippra/
│   ├── src/          (7 .c files, ~112 KB)
│   ├── include/rippra/ (7 headers)
│   ├── cuda/          (3 .cu kernels + header)
│   ├── ml/            (16 Python files — training, eval, benchmarking)
│   ├── tests/         (8 test/benchmark .c files)
│   ├── bindings/, tools/, viz/, scratch/, onnx_models/
│   └── build*.bat/.sh (9 platform-specific build scripts)
├── visualizations/    (static HTML/PNG, ~9 MB)
├── simulation_visualization/ (PNG dashboards)
└── .github/workflows/ci.yml
```

### Findings

| # | Finding | Severity | Detail |
|---|---|---|---|
| R1 | `rippra_api.c` duplicates `rippra_zonal_mesh`/`rippra_modal_model` struct definitions instead of including `recon.h`, because `rippra_api.h` and `recon.h` declare incompatible types under the same names | **High** | Confirmed by direct inspection of `rippra_api.c` lines 16–28. Any struct-layout change in `recon.h` will silently desync the public API. This is the single largest architectural risk in the codebase. |
| R2 | `CMakeLists.txt` calls `find_package(BLAS REQUIRED)` and links `${BLAS_LIBRARIES}`, but `la.c` explicitly documents itself as a **vendored, dependency-free** SVD/LU implementation | **Medium** | Either BLAS is dead weight (breaks builds on machines without BLAS, for no benefit) or there's an intended-but-unfinished BLAS acceleration path. Needs resolution one way or the other. |
| R3 | Nine separate `build_*.bat`/`build_*.sh` scripts alongside a working `CMakeLists.txt` | **Medium** | Duplicate, drifting build logic. Unclear which is authoritative; a contributor or judge cloning the repo has no single "how do I build this" answer. |
| R4 | 9 MB+ of committed PNG/GIF/HTML visualizations and a 2.8 MB `wavefront_3d_anim.gif` in the main tree | **Low** | Bloats clone size and git history; belongs in Git LFS, a releases artifact, or GitHub Pages build output, not the source tree. |
| R5 | `visualizations/index.html` is 1.9 MB of committed HTML | **Low** | Likely a bundled/pre-rendered dashboard. Should be a build artifact (CI-generated on release/Pages deploy), not hand-committed source. |
| R6 | No `.clang-format` / `.editorconfig` | **Low** | Contributor style is unenforced; `AGENTS.md`/`CONTRIBUTING.md` should reference a formatting standard. |
| R7 | `rippra/scratch/` directory committed to the main tree | **Low** | Scratch/experimental code should not ship in the audited deliverable tree; move to a branch or `.gitignore`. |

---

## 3. Problem Statement Compliance

| PS9 Requirement | Status | Evidence | Residual Gap |
|---|:---:|---|---|
| Centroid detection | ✅ | `rippa_compute_centroids` (TCoG), `rippa_compute_centroids_refined` (two-pass) | No unit test asserts sub-pixel accuracy against a synthetic spot with known sub-pixel offset |
| Spot deviation from reference | ✅ | `rippa_compute_deltas` | NaN handling zeroes deviation rather than masking the sub-aperture out of the reconstruction (see §12, S3) |
| Wavefront reconstruction (zonal, Fried geometry) | ✅ | `rippra_zonal_setup` + SVD pseudo-inverse in `la.c` | No documented residual/condition-number check on `Gpinv`; ill-conditioned geometries fail silently |
| Wavefront reconstruction (modal, Zernike) | ✅ (per `docs/problem_statement_alignment.md`) | `rippra_modal_reconstruct` | Not independently re-verified in this audit against a closed-form Zernike slope integral — recommend a golden-value test |
| Fried parameter r₀ | ✅ implemented | `rippra_compute_r0_impl` | No validation against a synthetic Kolmogorov screen of known r₀ |
| Coherence time τ₀ | ✅ implemented | 1/e auto-correlation decay with fractional-lag interpolation in `recon.c` | No validation against a synthetic time series of known τ₀; edge case where `C[0]==C[1]` (division by zero in fraction calc) not guarded |
| DM actuator mapping with coupling | ✅ | `rippra_dm_map_impl`, coupling + coupling² model in `recon.c` | Coupling model is a simple geometric nearest-neighbor rule, not a measured/fitted influence function — acceptable for a hackathon but should be labeled as a simplifying assumption in the paper/README |
| Real-time (<10 ms) | 🟡 claimed | README claims 761 µs end-to-end | Figure appears to be a **microbenchmark of the hot-path math only** (per README breakdown), not measured under the CI-built binary, not measured with I/O, not measured against the CUDA path — needs an actual end-to-end benchmark artifact, not a table in a doc |
| C implementation | ✅ | Core numerics entirely in C, OpenMP-parallel | — |
| Dataset compatibility (ISRO SH-WFS frames) | 🟡 | BMP/raw readers in `io.c`, `config/default.yaml` geometry assumptions | No evidence of a dry run against real/representative ISRO-format data; assumptions (12-bit depth, 1024×1024, 20 lenslets) are declared but not defended |

**Overall PS9 compliance: ~85% functionally implemented, ~55% independently validated.** The gap between "implemented" and "validated" is the single most important thing this audit surfaces, because ISRO evaluators will ask for exactly this evidence.

---

## 4–7. Research, Mathematics, Algorithm, and ML Audit (condensed)

- **Research coverage** is solid for the stated scope: Fried geometry, Zernike modal fitting, Kolmogorov-derived r₀, temporal-correlation τ₀, and a coupling-matrix DM model are all textbook-correct choices for PS9. Missing from the docs: any citation trail (Noll 1976, Fried 1965/1966, Hardy's *Adaptive Optics for Astronomical Telescopes*) tying implementation choices to literature — reviewers will want this in the paper, not just in code comments.
- **Mathematics**: the Zernike derivative evaluation in `recon.c` (`evaluate_zernike_derivatives`) is analytically implemented rather than finite-differenced, which is good practice, but has no unit test comparing it against a hand-verified closed-form value (e.g., Z₄ = defocus, dZ/dx at a known point). The one-sided Jacobi SVD in `la.c` has no documented convergence tolerance or iteration cap check reported back to the caller — silent non-convergence would return a garbage pseudo-inverse with no error code.
- **Algorithms** are within the accepted family used by AOtools/Soapy/HCIPy (interaction-matrix pseudo-inverse zonal reconstruction, Zernike modal reconstruction). No repository-side benchmark compares RIPRA's reconstruction RMS error against any of those references on a shared synthetic case — this comparison would meaningfully strengthen a paper/hackathon submission and is currently absent.
- **ML pipeline** (`ml/predictive_ao.py`, `sequence_models.py`, `ablation_study.py`, `noise_robustness.py`) is unusually mature for a hackathon side-project: has an LSTM predictive-AO path, ablations, and noise-robustness sweeps. README claims a CNN test MSE of 0.001957 (99.97% mean correlation) and a 6.6% RMS reduction from LSTM prediction under 1-frame latency. These numbers are **not reproducible from the repo as committed** — `ml_checkpoints/` and `data_ai/dataset.npz` are not present in the tree (confirmed absent from clone), and CI explicitly skips the ONNX/inference evaluation step "gracefully" when those artifacts are missing. A reviewer cannot currently regenerate the headline ML numbers from a fresh clone.

---

## 8–14. Performance, Dataset, Testing, Documentation, Visualization, Security, Code-Quality Audit (condensed)

- **Performance**: latency figures in the README are not backed by a checked-in benchmark log or CI performance gate. `tests/benchmark_centroid.c` and `tests/benchmark_openmp.c` exist but are not run in CI, so there's no historical record of whether performance regresses between commits.
- **Dataset**: `io.c`'s BMP reader trusts the 54-byte header blindly (`fread(hdr, 1, 54, fp)` checked, but subsequent header fields such as bit depth and stride are not validated against `config/default.yaml`'s stated 12-bit depth before use) — a malformed or differently-formatted ISRO file would silently misdecode rather than error out.
- **Testing**: 8 test files exist and are exercised for centroid/recon/io/la/stream/full-pipeline, which is genuinely good coverage for a hackathon repo. Missing: CUDA path (`test_cuda.c` is never run in CI — no GPU runner configured), closed-loop convergence test (`rippra_closed_loop_run_impl` has no dedicated test), and no golden-value numerical regression tests (i.e., "does r0 == 0.142 ± tolerance for this fixed synthetic input" checked into CI).
- **Documentation**: `docs/` is unusually thorough (11 files including a dedicated problem-statement alignment doc and a paper draft), which is a strength. Gaps: no `CHANGELOG.md` entries reflecting the CUDA/ML additions (it's 22 lines, likely stale relative to current tree size), no architecture diagram, no explicit "known limitations / simplifying assumptions" section anywhere (this matters a lot for a technical review — reviewers trust projects more when they state their own assumptions).
- **Visualization**: strong — 15+ static visualizations plus interactive HTML dashboards (`animated_wavefront.html`, `wavefront_3d_viewer.html`, `pipeline_dashboard.html`). Missing: no visualization ties reconstruction output to a *live* run of the actual binary (all appear to be pre-rendered/static per notebook headers).
- **Security/Reliability**: `fread(rowbuf, 1, stride, fp)` return value discarded in the BMP row-reading loop in `io.c` (line 156) — a truncated file would silently read garbage/zero-padded rows instead of erroring. `malloc` null-checks are present in most of `centroid.c`/`recon.c` but not verified across all of `rippra_api.c`. NaN centroids from lost/occluded spots are zeroed in `rippa_compute_deltas` rather than propagated as a "sub-aperture invalid" flag, meaning a dead spot contributes a false zero-displacement measurement to the reconstruction rather than being excluded — this is a genuine correctness bug under partial spot occlusion (cloud, saturation, vignetting at pupil edge), not just a style nit.
- **Code quality**: type duplication between `rippra_api.c` and `recon.h` (R1 above) is the standout SOLID violation. Otherwise the C code is consistently commented, uses consistent row-major conventions, and documents its coordinate system explicitly (a good practice too many optics codebases skip).

---

## 15. GitHub Issue Backlog

Below are actionable, copy-paste-ready issues grouped by category. Each is grounded in an actual, cited finding above (not a generic template).

### Category A — Correctness & Numerical Validation (Critical/High)

---
**A1. bug: NaN centroids silently injected as zero displacement, corrupting reconstruction**
- Labels: bug, high priority, adaptive-optics
- Priority: Critical
- Background: `rippa_compute_deltas` (`centroid.c`) sets `dx[i]=dy[i]=0.0` when a centroid is NaN (lost/occluded spot), instead of excluding that sub-aperture from the reconstruction.
- Problem Statement: A dead sub-aperture reports a false "zero slope" measurement, biasing both zonal and modal reconstruction and corrupting r₀/τ₀ estimates that depend on slope variance.
- Proposed Solution: Add a `valid` mask output from `rippa_compute_deltas`; propagate the mask into `rippra_zonal_reconstruct`/`rippra_modal_reconstruct` to exclude invalid rows from `G`/`Zprime` at reconstruction time, or re-weight via a diagonal weight matrix in the pseudo-inverse solve.
- Acceptance Criteria: [ ] mask propagated through centroid→recon API [ ] unit test with a synthetic frame containing ≥1 dead spot verifies reconstruction RMS is unaffected by the dead spot [ ] docs updated [ ] CHANGELOG entry
- References: Hardy, *Adaptive Optics for Astronomical Telescopes*, ch. 5 (bad-actuator/bad-subaperture masking)
- Dependencies: none
- Estimated Difficulty: M
- Estimated Time: 1–2 days

---
**A2. fix: resolve struct-definition duplication between rippra_api.c and recon.h**
- Labels: bug, high priority, refactor
- Priority: Critical
- Background: `rippra_api.c` cannot include `recon.h` because `rippra_api.h` and `recon.h` declare incompatible types with the same names, so it hand-duplicates `rippra_zonal_mesh`/`rippra_modal_model` struct layouts.
- Problem Statement: Any future change to these structs in `recon.h` will silently desynchronize from the copies in `rippra_api.c`, producing memory corruption with no compiler warning.
- Proposed Solution: Rename the conflicting types in `rippra_api.h` (e.g. `rippra_api_zonal_mesh`) or unify both headers on one canonical set of types with `_impl` suffix functions operating on the same structs; remove the duplicated definitions.
- Acceptance Criteria: [ ] single source of truth for struct layout [ ] `rippra_api.c` includes `recon.h` directly [ ] full test suite passes [ ] static assert or `sizeof` check added to CI to catch future desync
- References: —
- Dependencies: none
- Estimated Difficulty: M
- Estimated Time: 1 day

---
**A3. test: add golden-value regression tests for r0, tau0, and Zernike derivatives**
- Labels: testing, mathematics, good first issue
- Priority: High
- Background: r₀/τ₀/Zernike-derivative code paths run without any test asserting numerical correctness against a hand-computed or literature reference value.
- Problem Statement: "Runs without crashing" is currently the only test bar; no test would catch a sign error, unit error, or normalization error in r₀/τ₀/Zernike math.
- Proposed Solution: Add fixed synthetic inputs with independently pre-computed expected outputs (e.g., Z₄ defocus derivative at ρ=0.5,θ=0; r₀ from a fixed slope-variance input using the closed-form Kolmogorov relation) and assert within tight tolerance.
- Acceptance Criteria: [ ] ≥5 golden-value tests added [ ] tests run in CI [ ] tolerance documented and justified
- References: Noll, "Zernike polynomials and atmospheric turbulence," JOSA 1976
- Dependencies: none
- Estimated Difficulty: M
- Estimated Time: 2 days

---
**A4. bug: guard division-by-zero in tau0 fractional-lag interpolation**
- Labels: bug, mathematics
- Priority: High
- Background: In `rippra_compute_tau0_impl`-adjacent code, `fraction = (target - C_prev) / (C_curr - C_prev)` has no guard against `C_curr == C_prev`.
- Problem Statement: A flat or degenerate auto-correlation curve (e.g., near-zero turbulence, or a too-short time series) divides by zero, producing NaN/inf τ₀ silently.
- Proposed Solution: Guard the denominator; if degenerate, fall back to nearest-lag τ₀ estimate and flag low-confidence in an output diagnostic.
- Acceptance Criteria: [ ] guard added [ ] unit test with flat correlation input [ ] no NaN/inf can propagate to API output
- References: —
- Dependencies: A3
- Estimated Difficulty: XS
- Estimated Time: 2 hours

---
**A5. feat: validate zonal/modal reconstruction against synthetic Kolmogorov screens of known statistics**
- Labels: research, testing, high priority
- Priority: High
- Background: No test currently drives a full synthetic Kolmogorov phase screen of known r₀ through the entire centroid→reconstruct→r0/tau0 pipeline and checks the recovered r₀ against the input.
- Problem Statement: Without end-to-end validation, correctness of the full chain is unproven even if individual functions pass unit tests.
- Proposed Solution: Generate (or reuse `ml/synthetic_shwfs.py`) a screen with a specified r₀, feed through the C pipeline, assert recovered r₀ within X% of ground truth across several turbulence strengths.
- Acceptance Criteria: [ ] end-to-end validation script added [ ] tolerance table across weak/moderate/strong turbulence [ ] results committed as a documented benchmark, not just console output
- References: Roddier, *Adaptive Optics in Astronomy*, ch. 2
- Dependencies: A1, A3
- Estimated Difficulty: L
- Estimated Time: 3–4 days

---
**A6. fix: check fread return value in BMP row reader**
- Labels: bug, security
- Priority: Medium
- Background: `io.c` line 156, `fread(rowbuf, 1, stride, fp)` inside the BMP row loop discards the return value.
- Problem Statement: A truncated or corrupt BMP file (e.g. partial download, disk error) silently reads short/zero-padded rows rather than erroring, producing a plausible-looking but wrong frame.
- Proposed Solution: Check `fread` return equals `stride`; return an explicit I/O error code otherwise.
- Acceptance Criteria: [ ] return value checked [ ] test with a truncated BMP fixture asserts error code returned, not silent success
- References: —
- Dependencies: none
- Estimated Difficulty: XS
- Estimated Time: 2 hours

---
**A7. fix: validate BMP header fields (bit depth, dimensions) against config before decode**
- Labels: bug, security
- Priority: Medium
- Background: The BMP reader consumes the 54-byte header but does not appear to cross-check bit depth/dimensions against `config/default.yaml`'s declared camera geometry before decoding pixel data.
- Problem Statement: A mismatched or malformed file (wrong bit depth, unexpected stride) would misdecode rather than fail cleanly — a real risk when ingesting an externally supplied ISRO dataset.
- Proposed Solution: Add explicit validation with descriptive error return codes for bit-depth/dimension mismatch.
- Acceptance Criteria: [ ] validation added [ ] test fixtures for mismatched bit depth and wrong dimensions both return clean errors
- References: —
- Dependencies: A6
- Estimated Difficulty: S
- Estimated Time: half day

---
**A8. feat: report SVD non-convergence / condition number from la.c pseudo-inverse**
- Labels: mathematics, bug, robustness
- Priority: Medium
- Background: The one-sided Jacobi SVD in `la.c` has no visible mechanism to report iteration-cap exhaustion or an ill-conditioned input matrix to the caller.
- Problem Statement: A geometry that produces an ill-conditioned interaction matrix (e.g. very few active sub-apertures, degenerate spot layout) would silently return a garbage pseudo-inverse.
- Proposed Solution: Return convergence status and condition number estimate (ratio of max/min retained singular value) from `rippa_pinv`; propagate as a warning/error code up through `rippra_zonal_setup`/`rippra_modal_setup`.
- Acceptance Criteria: [ ] convergence/condition number surfaced in API [ ] test with a deliberately ill-conditioned matrix triggers the warning path
- References: —
- Dependencies: none
- Estimated Difficulty: M
- Estimated Time: 1–2 days

### Category B — Testing & CI (High/Medium)

---
**B1. ci: add CUDA build and test job**
- Labels: performance, testing, CI/CD
- Priority: High
- Background: `rippra/cuda/*.cu` and `tests/test_cuda.c` exist but no CI job builds or runs them.
- Problem Statement: The CUDA path is entirely unverified in CI; regressions there would go undetected indefinitely.
- Proposed Solution: Add a self-hosted or GPU-enabled CI job (e.g. a GitHub-hosted GPU runner, or a nightly job on available hardware) building with `nvcc` and running `test_cuda`; alternatively, add a CPU-only "compiles cleanly" job as a minimum bar if GPU runners aren't available.
- Acceptance Criteria: [ ] CUDA compile step in CI [ ] test_cuda runs (or, if no GPU runner, at minimum a syntax/compile-only check) [ ] badge in README
- References: —
- Dependencies: none
- Estimated Difficulty: M
- Estimated Time: 1–2 days

---
**B2. ci: run performance benchmarks in CI and track regressions**
- Labels: performance, CI/CD
- Priority: Medium
- Background: `tests/benchmark_centroid.c` and `tests/benchmark_openmp.c` exist but are not run in `ci.yml`.
- Problem Statement: README latency claims (e.g., 761 µs end-to-end) have no CI-verified provenance; performance regressions between commits are invisible.
- Proposed Solution: Add a CI step running the benchmarks and uploading results as a build artifact; optionally fail the build if latency regresses beyond a threshold.
- Acceptance Criteria: [ ] benchmark step added to `ci.yml` [ ] results uploaded as artifact per run [ ] README latency table links to a specific benchmark artifact/run, not just an unlinked number
- References: —
- Dependencies: none
- Estimated Difficulty: S
- Estimated Time: 1 day

---
**B3. test: add closed-loop convergence test for rippra_closed_loop_run_impl**
- Labels: testing, good first issue
- Priority: Medium
- Background: `recon.c` implements a full closed-loop AO iteration (`rippra_closed_loop_run_impl`) with gain, max-iter, and target-RMS parameters, but no dedicated test drives it to convergence.
- Problem Statement: Closed-loop stability/convergence behavior (a headline PS9-adjacent capability) is unverified.
- Proposed Solution: Add a test with a fixed initial phase and gain, assert convergence within N iterations and residual RMS below target; add a divergence case (gain too high) asserting the function returns the max-iter code rather than hanging or crashing.
- Acceptance Criteria: [ ] convergent-case test [ ] divergent-case test [ ] both run in CI
- References: —
- Dependencies: none
- Estimated Difficulty: S
- Estimated Time: 1 day

---
**B4. test: reproduce README ML accuracy claims from a fresh clone**
- Labels: testing, machine-learning, documentation
- Priority: High
- Background: README cites CNN test MSE 0.001957 / 99.97% correlation and a 6.6% LSTM RMS improvement, but `ml_checkpoints/` and `data_ai/dataset.npz` are absent from the repo, and CI explicitly skips evaluation when they're missing.
- Problem Statement: A reviewer or judge cannot currently reproduce the headline ML numbers from `git clone` alone — this is a credibility risk for both the hackathon submission and any paper draft citing these numbers.
- Proposed Solution: Add a CI job (or documented one-command script) that generates the synthetic dataset via `ml/synthetic_shwfs.py`, trains (or loads a committed small checkpoint via Git LFS/release asset), and reproduces the reported metrics within tolerance; otherwise soften README claims to "measured on internal training run — reproduction script: `ml/train.py`."
- Acceptance Criteria: [ ] either full repro pipeline runs in CI, or README is updated with explicit reproduction instructions and caveats [ ] no unreproducible number remains stated as fact without a reproduction path
- References: —
- Dependencies: none
- Estimated Difficulty: L
- Estimated Time: 2–3 days

---
**B5. test: add fuzz/malformed-input tests for io.c readers**
- Labels: testing, security
- Priority: Medium
- Background: BMP/raw readers in `io.c` handle well-formed input paths but have no tests for truncated, oversized, or corrupted files.
- Problem Statement: Given this pipeline will eventually ingest externally supplied ISRO sensor data, malformed-input robustness is a real (not hypothetical) requirement.
- Proposed Solution: Add a small corpus of malformed BMP/raw fixtures (truncated, wrong magic bytes, zero dimensions, huge dimensions) and assert clean error returns, no crashes, no unbounded allocation.
- Acceptance Criteria: [ ] ≥6 malformed fixtures [ ] all return clean error codes [ ] run under a memory sanitizer in CI (ASan) for this test binary specifically
- References: —
- Dependencies: A6, A7
- Estimated Difficulty: M
- Estimated Time: 1–2 days

---
**B6. ci: build and run under AddressSanitizer / UndefinedBehaviorSanitizer**
- Labels: security, CI/CD, good first issue
- Priority: Medium
- Background: No sanitizer-instrumented build exists in CI despite heavy manual memory management (`malloc`/`free`, raw pointer arithmetic in `centroid.c`/`recon.c`/`la.c`).
- Problem Statement: Memory bugs (use-after-free, leaks on error paths, out-of-bounds in the connected-components labeling) are currently only discoverable by manual review.
- Proposed Solution: Add a CI job compiling with `-fsanitize=address,undefined` and running the full test suite under it.
- Acceptance Criteria: [ ] sanitizer job added [ ] passes clean, or documented known issues filed as follow-up bugs
- References: —
- Dependencies: none
- Estimated Difficulty: S
- Estimated Time: half day

---
**B7. ci: add static analysis (cppcheck / clang-tidy) gate**
- Labels: code-quality, CI/CD, good first issue
- Priority: Low
- Background: No static analysis tool runs in CI.
- Problem Statement: Style/quality drift and common C pitfalls (e.g. the discarded `fread` return in A6) go uncaught between reviews.
- Proposed Solution: Add `cppcheck` and/or `clang-tidy` as a non-blocking (initially) CI job; tighten to blocking once the codebase passes clean.
- Acceptance Criteria: [ ] job added [ ] baseline report committed [ ] plan to reach zero-warnings documented in CONTRIBUTING.md
- References: —
- Dependencies: none
- Estimated Difficulty: S
- Estimated Time: half day

### Category C — Architecture & Code Quality (Medium)

---
**C1. refactor: consolidate 9 build scripts into a single CMake-driven build**
- Labels: refactor, good first issue
- Priority: Medium
- Background: `build.bat`, `build.sh`, `build_benchmark.bat`, `build_cuda.sh`, `build_cuda_test.bat`, `build_dll.bat`, `build_openmp.bat`, `build_test_pipeline.bat`, `build_test_recon.bat`, `build_test_stream.bat` coexist with a working `CMakeLists.txt`.
- Problem Statement: Drift between these scripts and CMake is a maintenance and correctness risk; new contributors don't know which is authoritative.
- Proposed Solution: Migrate all build variants (benchmark, CUDA, DLL, OpenMP, per-test) into CMake options/targets (`-DRIPRA_BUILD_CUDA=ON`, `-DRIPRA_BUILD_BENCHMARKS=ON`, etc.); deprecate and remove the standalone scripts once parity is confirmed.
- Acceptance Criteria: [ ] all current script capabilities available as CMake options [ ] scripts removed or reduced to thin CMake wrappers [ ] docs/build_guide.md updated
- References: —
- Dependencies: none
- Estimated Difficulty: L
- Estimated Time: 2–3 days

---
**C2. build: resolve BLAS dependency inconsistency in CMakeLists.txt**
- Labels: build, good first issue
- Priority: Medium
- Background: `CMakeLists.txt` requires and links BLAS; `la.c` documents itself as a vendored, dependency-free SVD/LU implementation and does not appear to call BLAS routines.
- Problem Statement: Unclear whether BLAS is a dead requirement (breaks builds unnecessarily on minimal systems) or an unfinished acceleration path.
- Proposed Solution: Either remove the BLAS dependency entirely, or add an actual BLAS-backed fast path for the LU/matmul routines with a documented fallback to the vendored implementation.
- Acceptance Criteria: [ ] decision documented in CMakeLists.txt comment [ ] build succeeds on a machine with no BLAS installed if dependency is removed, or BLAS path is actually exercised if kept
- References: —
- Dependencies: none
- Estimated Difficulty: S
- Estimated Time: half day

---
**C3. chore: move committed visualization binaries out of the source tree**
- Labels: repository-hygiene, good first issue
- Priority: Low
- Background: `visualizations/` and `simulation_visualization/` contain ~10 MB of PNG/GIF/HTML committed directly.
- Problem Statement: Bloats clone size/history; these look like generated artifacts, not source.
- Proposed Solution: Move to Git LFS, or regenerate via a documented script and publish via GitHub Pages/Releases instead of committing binaries.
- Acceptance Criteria: [ ] repo size reduced [ ] regeneration script documented [ ] Pages or release-asset publishing configured if these are meant to be user-facing
- References: —
- Dependencies: none
- Estimated Difficulty: S
- Estimated Time: 1 day

---
**C4. chore: remove or relocate rippra/scratch/ from the audited tree**
- Labels: repository-hygiene, good first issue
- Priority: Low
- Background: A `scratch/` directory is committed directly under `rippra/`.
- Problem Statement: Experimental/scratch code in the main tree is confusing for reviewers trying to identify the deliverable surface.
- Proposed Solution: Move to a separate branch, or `.gitignore` it and document the convention in CONTRIBUTING.md.
- Acceptance Criteria: [ ] scratch/ removed from main branch tree or clearly marked non-deliverable in README
- References: —
- Dependencies: none
- Estimated Difficulty: XS
- Estimated Time: 1 hour

---
**C5. docs: add .clang-format and enforce in CI**
- Labels: code-quality, good first issue
- Priority: Low
- Background: No formatting config exists; style consistency currently depends on manual discipline.
- Proposed Solution: Add `.clang-format` matching current style conventions (row-major, brace style observed in `centroid.c`/`recon.c`), add a CI check step.
- Acceptance Criteria: [ ] `.clang-format` added [ ] CI check job added [ ] existing code reformatted in one clean commit
- References: —
- Dependencies: none
- Estimated Difficulty: S
- Estimated Time: half day

---
**C6. refactor: add explicit "known simplifying assumptions" module/doc surfaced from config**
- Labels: documentation, research
- Priority: Medium
- Background: The DM coupling model (`coupling`, `coupling²` nearest/diagonal-neighbor rule in `recon.c`) is a simplifying assumption, not a measured influence function; this is reasonable for a hackathon but is not labeled as such anywhere visible to a reviewer.
- Proposed Solution: Add a "Known Assumptions & Simplifications" section to the README/paper explicitly listing: geometric (not measured) DM coupling, synthetic (not yet real-dataset-validated) turbulence statistics, microbenchmark-only latency figures.
- Acceptance Criteria: [ ] section added to README and paper draft [ ] each assumption cross-referenced to the code location that encodes it
- References: —
- Dependencies: none
- Estimated Difficulty: XS
- Estimated Time: 2 hours

### Category D — Machine Learning (Medium/High)

---
**D1. feat: commit or auto-generate reproducible ML training artifacts**
- Labels: machine-learning, high priority
- Priority: High
- Background: See B4 — `ml_checkpoints/` and `data_ai/dataset.npz` are absent from the repo; CI skips ML evaluation gracefully when missing.
- Problem Statement: Duplicate of B4's root cause from the ML side — no reviewer can currently verify the ML claims end-to-end.
- Proposed Solution: Either commit small checkpoints via Git LFS/GitHub Releases, or make `ml/synthetic_shwfs.py` → `ml/train.py` → `ml/evaluate_inference.py` a documented one-command reproducible pipeline runnable in CI within time budget.
- Acceptance Criteria: [ ] one-command reproduction documented in `ml/README.md` [ ] CI runs it end-to-end on at least a small/fast config
- References: —
- Dependencies: B4
- Estimated Difficulty: L
- Estimated Time: 2–3 days

---
**D2. feat: add train/validation/test split leakage audit for sequence models**
- Labels: machine-learning
- Priority: Medium
- Background: `sequence_models.py`/`train_sequence.py` train an LSTM on temporal Zernike sequences; temporal data is notoriously easy to leak across train/test splits (adjacent frames in test appearing in train).
- Problem Statement: No visible documentation of how splits are constructed for time-series data; if split is random-frame rather than contiguous-block, reported accuracy is likely inflated by leakage.
- Proposed Solution: Document (or fix) the split strategy to use contiguous time blocks; add an explicit leakage-check test (assert no temporal overlap between train/val/test index sets).
- Acceptance Criteria: [ ] split strategy documented [ ] leakage-check test added [ ] any affected reported metrics re-measured and corrected if needed
- References: Bergmeir & Benítez, "On the use of cross-validation for time series," 2012
- Dependencies: none
- Estimated Difficulty: M
- Estimated Time: 1–2 days

---
**D3. feat: report inference latency of ONNX models under realistic deployment constraints**
- Labels: machine-learning, performance
- Priority: Medium
- Background: `ml/benchmark_gpu.py` and `evaluate_inference.py` exist, but latency numbers, if reported, should be re-measured on the CI runner or a documented representative embedded/edge target, not just a dev GPU.
- Proposed Solution: Add a benchmark report comparing ONNX CPU vs GPU inference latency, explicitly stating hardware, batch size, and whether this fits inside the <10 ms real-time budget when the ML path is in the loop (vs. the classical zonal/modal path).
- Acceptance Criteria: [ ] benchmark documented with hardware spec [ ] explicit statement of whether ML-in-the-loop still meets the 10 ms constraint
- References: —
- Dependencies: none
- Estimated Difficulty: S
- Estimated Time: 1 day

---
**D4. research: justify (or explicitly decline) Transformer/PINN/ViT/GNN extensions**
- Labels: research, machine-learning, low priority
- Priority: Low
- Background: Phase-6-style audits often ask "why not Transformer/PINN/ViT/GNN." Given the current MLP/CNN/LSTM stack already fits the temporal Zernike-prediction problem well and the real-time budget is tight, larger architectures are not obviously justified.
- Proposed Solution: Add a short "Architecture Choice Justification" note explaining that the sequence length and mode count here don't benefit from attention/graph structure the way large-aperture segmented-mirror AO does, and that inference-latency budget rules out heavier architectures — closing this line of inquiry deliberately rather than leaving it open.
- Acceptance Criteria: [ ] short justification note added to `docs/algorithms.md` or paper
- References: —
- Dependencies: none
- Estimated Difficulty: XS
- Estimated Time: 1 hour

### Category E — Documentation (Medium/Low)

---
**E1. docs: add architecture diagram (data flow: sensor frame → centroid → recon → r0/tau0 → DM)**
- Labels: documentation, good first issue
- Priority: Medium
- Background: 11 documentation files exist but none is a single visual data-flow diagram of the whole pipeline.
- Proposed Solution: Add one diagram (Mermaid or SVG) to the README showing frame ingestion → calibration → per-frame centroiding → zonal/modal reconstruction → r0/tau0 → DM mapping → closed-loop, with C module names annotated on each stage.
- Acceptance Criteria: [ ] diagram added and referenced from README top [ ] matches actual function names in code
- References: —
- Dependencies: none
- Estimated Difficulty: S
- Estimated Time: half day

---
**E2. docs: refresh CHANGELOG.md to reflect CUDA/ML/testing additions**
- Labels: documentation, good first issue
- Priority: Low
- Background: `CHANGELOG.md` is 22 lines; the tree now includes CUDA kernels, a 16-file ML pipeline, and 8 test files that are unlikely to all be reflected.
- Proposed Solution: Backfill CHANGELOG entries for major additions; adopt Keep-a-Changelog format going forward.
- Acceptance Criteria: [ ] CHANGELOG reflects current tree contents [ ] format documented in CONTRIBUTING.md
- References: —
- Dependencies: none
- Estimated Difficulty: XS
- Estimated Time: 2 hours

---
**E3. docs: add literature citation list to docs/mathematical_foundation.md**
- Labels: documentation, research
- Priority: Medium
- Background: Mathematical derivations (Zernike, r0, tau0) are implemented and lightly commented in code but not tied to a citation list in the docs.
- Proposed Solution: Add citations for Noll 1976 (Zernike/turbulence), Fried 1965/1966 (Fried geometry/r0), Southwell 1980 (comparison point for zonal geometries), Hardy's AO textbook.
- Acceptance Criteria: [ ] citation list added [ ] each major formula in the doc links to its source
- References: as listed above
- Dependencies: none
- Estimated Difficulty: XS
- Estimated Time: 2 hours

---
**E4. docs: add API reference examples for the public C API (rippra_api.h)**
- Labels: documentation, good first issue
- Priority: Low
- Background: `docs/api_reference.md` exists (8 KB) but should be checked for completeness against the actual public surface in `rippra_api.h` given the type-duplication issue in A2 makes the "public API" and "internal recon API" easy to conflate.
- Proposed Solution: Add a minimal worked example (calibrate → compute centroids → reconstruct → get r0/tau0 → get DM commands) using only the public API, compiled and tested as a doc-example in CI.
- Acceptance Criteria: [ ] worked example added [ ] compiled/run in CI as a smoke test
- References: —
- Dependencies: A2
- Estimated Difficulty: S
- Estimated Time: 1 day

---
**E5. docs: state real-dataset compatibility assumptions explicitly**
- Labels: documentation, dataset
- Priority: High
- Background: `config/default.yaml` encodes specific camera/MLA/DM parameters (1024×1024, 12-bit, 150 µm pitch, 20 lenslets, 21 actuators) presented as defaults without an explicit statement of which are placeholders vs. which are derived from the actual ISRO problem statement/dataset spec.
- Proposed Solution: Annotate `config/default.yaml` (and a corresponding docs section) marking each parameter as "from official PS9 spec" vs. "assumed placeholder, override before real submission."
- Acceptance Criteria: [ ] every config parameter annotated with provenance [ ] doc section added explaining override procedure once real dataset spec/sample is available
- References: —
- Dependencies: none
- Estimated Difficulty: XS
- Estimated Time: 2 hours

### Category F — GitHub Project Hygiene (Low/Medium)

---
**F1. chore: add issue and pull request templates**
- Labels: good first issue, repository-hygiene
- Priority: Medium
- Proposed Solution: Add `.github/ISSUE_TEMPLATE/bug_report.yml`, `feature_request.yml`, and `.github/PULL_REQUEST_TEMPLATE.md`.
- Acceptance Criteria: [ ] templates added [ ] linked from CONTRIBUTING.md
- Estimated Difficulty: XS
- Estimated Time: 1 hour

---
**F2. chore: define a labels taxonomy and apply consistently**
- Labels: good first issue, repository-hygiene
- Priority: Low
- Proposed Solution: Create a `labels.yml`/use `gh label create` script to define the category labels used throughout this backlog (bug, enhancement, documentation, performance, research, testing, machine-learning, security, code-quality, repository-hygiene, good first issue, CI/CD, high priority, critical) consistently.
- Acceptance Criteria: [ ] label set created in repo [ ] backlog issues (this document) filed with matching labels
- Estimated Difficulty: XS
- Estimated Time: 1 hour

---
**F3. ci: add CodeQL / dependency review scanning**
- Labels: security, CI/CD
- Priority: Medium
- Proposed Solution: Add GitHub's CodeQL workflow for C and Python, plus Dependabot config for the Python ML dependencies.
- Acceptance Criteria: [ ] CodeQL workflow added and passing [ ] Dependabot config added
- Estimated Difficulty: S
- Estimated Time: half day

---
**F4. chore: set up a GitHub Project board with the milestones in §16**
- Labels: repository-hygiene
- Priority: Medium
- Proposed Solution: Create a Project (v2) board with columns Backlog/In Progress/Review/Done, and add all issues from this document tagged with the milestone they belong to.
- Acceptance Criteria: [ ] board created [ ] all issues assigned to a milestone and column
- Estimated Difficulty: XS
- Estimated Time: 1 hour

---
**F5. docs: add SECURITY.md coordinated-disclosure contact verification**
- Labels: security, documentation
- Priority: Low
- Background: `SECURITY.md` exists; verify it names a real contact/process rather than boilerplate, given this repo will be publicly linked from a hackathon/paper submission.
- Acceptance Criteria: [ ] contact verified functional [ ] response-time expectation stated
- Estimated Difficulty: XS
- Estimated Time: 30 minutes

---
**F6. docs: add CODEOWNERS**
- Labels: repository-hygiene, good first issue
- Priority: Low
- Proposed Solution: Add `.github/CODEOWNERS` mapping `rippra/src/`, `rippra/cuda/`, `rippra/ml/`, `docs/` to responsible reviewers.
- Acceptance Criteria: [ ] CODEOWNERS added
- Estimated Difficulty: XS
- Estimated Time: 30 minutes

### Category G — Performance & Real-Time Hardening (Medium)

---
**G1. perf: add end-to-end (not microbenchmark-only) latency measurement including I/O**
- Labels: performance, high priority
- Priority: High
- Background: README's 761 µs figure appears to cover hot-path math only (per the breakdown table), excluding frame ingestion/I/O and any GPU transfer overhead.
- Proposed Solution: Add a benchmark that measures wall-clock time from "frame available" to "DM command issued," including I/O and any host↔device transfer if the CUDA path is used, and report both classical and CUDA-path numbers.
- Acceptance Criteria: [ ] end-to-end benchmark added [ ] README updated with clearly labeled "hot-path only" vs "end-to-end" numbers
- Dependencies: B2
- Estimated Difficulty: M
- Estimated Time: 1–2 days

---
**G2. perf: add SIMD/AVX vectorization to the TCoG inner loop**
- Labels: performance, optimization
- Priority: Medium
- Background: `tcog_window`/`tcog_window_fast` in `centroid.c` are scalar per-pixel loops; this is the dominant per-frame hot path.
- Proposed Solution: Add an AVX2 (or portable SIMD via compiler auto-vectorization hints/`-O3 -march=native`) variant with a scalar fallback; benchmark before/after.
- Acceptance Criteria: [ ] SIMD variant added with runtime feature detection [ ] benchmark shows measurable speedup [ ] scalar fallback preserved for portability
- Dependencies: G1
- Estimated Difficulty: M
- Estimated Time: 2 days

---
**G3. perf: profile and document OpenMP scaling behavior across thread counts**
- Labels: performance, good first issue
- Priority: Low
- Background: OpenMP pragmas exist throughout `centroid.c`/`la.c`/`recon.c` but no committed scaling study (1/2/4/8 threads) exists.
- Proposed Solution: Add a scaling benchmark and plot/table to `docs/`.
- Acceptance Criteria: [ ] scaling data committed for at least 4 thread counts [ ] documented in performance docs
- Dependencies: B2
- Estimated Difficulty: S
- Estimated Time: 1 day

---
**G4. perf: benchmark CUDA path against CPU/OpenMP path on identical input**
- Labels: performance, CI/CD
- Priority: Medium
- Background: CUDA kernels exist (`centroid_kernels.cu`, `dm_kernels.cu`, `matrix_kernels.cu`) but no committed apples-to-apples comparison against the CPU path exists.
- Proposed Solution: Run identical synthetic input through both paths, report speedup, and note the crossover problem size (if any) below which CPU-only is actually faster due to transfer overhead.
- Acceptance Criteria: [ ] comparison benchmark committed [ ] crossover point documented if applicable
- Dependencies: B1
- Estimated Difficulty: M
- Estimated Time: 1–2 days

### Category H — Visualization (Low)

---
**H1. feat: generate at least one visualization directly from a live binary run, not pre-rendered**
- Labels: visualization
- Priority: Medium
- Background: Existing visualizations appear pre-rendered from notebooks rather than generated by invoking the actual compiled C pipeline.
- Proposed Solution: Add a small script/CLI flag that runs the real `librippra` pipeline on a synthetic frame and emits a live wavefront-surface plot, proving the visualization reflects the shipped code rather than a notebook prototype.
- Acceptance Criteria: [ ] live-generated visualization added [ ] clearly labeled as such vs. the notebook-derived ones
- Estimated Difficulty: M
- Estimated Time: 1–2 days

---
**H2. feat: add a DM actuator stroke heatmap visualization**
- Labels: visualization, good first issue
- Priority: Low
- Background: Existing visualizations cover wavefront/turbulence/Zernike well but no dedicated DM actuator command heatmap was found.
- Proposed Solution: Add a heatmap plotting `dm_commands` over the actuator grid (`node_u`/`node_v`) for a representative frame.
- Acceptance Criteria: [ ] heatmap added to visualization set
- Estimated Difficulty: S
- Estimated Time: half day

---
**H3. feat: add a performance dashboard tying latency benchmarks (G1–G4) to a live chart**
- Labels: visualization, performance
- Priority: Low
- Proposed Solution: Extend `pipeline_dashboard.html` to plot the CI-tracked benchmark history from B2/G4 rather than a static number.
- Acceptance Criteria: [ ] dashboard reads from committed benchmark artifacts
- Dependencies: B2, G4
- Estimated Difficulty: M
- Estimated Time: 1 day

---

## 16. Milestones / Project Board

| Milestone | Scope | Key Issues |
|---|---|---|
| **M1 — Correctness Hardening** | Fix the real bugs before anything else | A1, A2, A4, A6, A7, A8 |
| **M2 — Validation** | Prove the numbers are right, not just present | A3, A5, D2 |
| **M3 — CI/Test Breadth** | Cover CUDA, ML, closed-loop, fuzz, sanitizers | B1–B7 |
| **M4 — Architecture & Hygiene Cleanup** | Build system, repo bloat, formatting | C1–C6, F1–F6 |
| **M5 — Performance Proof** | Real end-to-end numbers, SIMD, CUDA comparison | G1–G4 |
| **M6 — Submission Polish** | Docs, visualizations, ML reproducibility, assumptions stated | D1, D3, D4, E1–E5, H1–H3 |

---

## 17. Development Roadmap

- **Immediate (this week):** A1, A2, A4, A6 — these are the correctness bugs a judge or careful reviewer would find fastest, and they're each ≤2 days.
- **Next week:** A3, A5, A8, B3, B5, B6 — validation tests and sanitizer/fuzz coverage, so the "it works" claim has evidence behind it.
- **Next month:** B1, B2, B4, D1, D2, C1, C2, F1–F4 — CI breadth, ML reproducibility, build consolidation, GitHub hygiene.
- **Pre-submission:** G1–G4 (real end-to-end performance numbers, since "10 ms real-time" is a headline PS9 criterion and needs to survive scrutiny), E5 (dataset assumptions stated), C6 (assumptions documented).
- **Post-submission / research publication track:** D3, D4, E3, H1–H3 — the kind of polish that matters more for an SN Computer Science-style paper than for the hackathon deadline itself.

---

## 18. Final Scorecard (0–10)

| Category | Score | Rationale |
|---|:---:|---|
| Scientific Correctness | 7 | Core math is standard and appears implemented correctly on inspection; unproven by dedicated golden-value/end-to-end validation tests |
| Software Engineering | 6 | Real modular C library with a working build, but struct-duplication landmine (A2) and 9 redundant build scripts (C1) |
| Documentation | 8 | Unusually thorough for a hackathon repo — 11 docs files, dedicated compliance mapping doc, paper draft |
| Mathematics | 7 | Zernike/Fried/Kolmogorov/coupling models are textbook-appropriate; missing convergence/condition reporting (A8) |
| AI/ML | 6 | Mature pipeline breadth (MLP/CNN/LSTM/ablation/noise-robustness), but headline metrics not reproducible from a fresh clone (D1/B4) |
| Performance | 6 | OpenMP + CUDA present; latency claims not independently, end-to-end verified (G1) |
| Real-Time Capability | 6 | Architecturally sound for <10 ms, but the actual end-to-end number (vs. hot-path-only microbenchmark) is unverified |
| Visualization | 8 | Strong breadth of static and interactive visualizations |
| Research Depth | 7 | Correct concepts, missing explicit literature citation trail (E3) |
| Innovation | 6 | Predictive-AO LSTM and coupling-based DM model are reasonable, not groundbreaking, choices — appropriate for the problem |
| Reproducibility | 5 | Weakest category — ML artifacts and some benchmark numbers cannot currently be regenerated from `git clone` alone |
| Maintainability | 6 | Good in-code documentation and naming; undermined by struct duplication (A2) and build-script sprawl (C1) |

**Composite: 6.8 / 10**

---

## 19. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| ISRO evaluators ask to reproduce the ML/latency numbers and can't from the public repo | High | High | D1, B4, G1 |
| A hidden test-set-vs-real-dataset geometry mismatch surfaces only at submission time | Medium | High | A7, E5 |
| Struct-duplication bug (A2) causes a crash under a future contributor's change, discovered late | Medium | High | A2, address before any further feature work on `rippra_api.c` |
| Reviewer discounts the submission for unlabeled simplifying assumptions (DM coupling model) looking like an oversight rather than a deliberate choice | Medium | Medium | C6, E5 |
| Undiscovered memory bug in the C core surfaces during a live demo | Low–Medium | High | B6 (sanitizers), B5 (fuzz) |

---

## 20. Final Verdict

🟢 **Submit-capable with a focused hardening pass, not a rewrite.** The prior stale audit's "do not submit" verdict no longer reflects the codebase. The core PS9 deliverables are implemented with reasonable algorithmic choices. The highest-leverage work between now and submission is not new features — it's (1) fixing the handful of real correctness bugs in Category A, (2) proving the numbers with validation tests and end-to-end benchmarks rather than asserting them in docs, and (3) making the ML claims reproducible from a clean clone. Everything in Categories C, F, and H is polish that helps a reviewer trust the project but won't change whether the core algorithm works.
