# Project RIPRA (ऋप्र)

<p align="center">
  <img src="./visualizations/logo.png" width="300" alt="Project RIPRA Logo"/>
</p>

Developing and optimizing algorithms for **Wavefront Reconstruction** and **Turbulence Characterization** using Shack-Hartmann Wavefront Sensor (SH-WFS) time-series data.

> **Note:** For a comprehensive analysis of the project results, detailed mathematical formulations, and the complete gallery of all 11 simulation telemetry plots, please refer to our [GitHub Discussion: End-to-End Wavefront Reconstruction & Control Benchmarks](https://github.com/PxA-Labs/Project-RIPRA/discussions).

---

## Description
Turbulence in the atmosphere distorts a plane-parallel wavefront propagating through it. A Shack-Hartmann Wavefront Sensor (SH-WFS) samples this distorted wavefront using an array of small lenslets (Microlens Array - MLA). The MLA creates a spot-field on the camera detector, and the spatial deviation of these spots from their reference positions is used to derive the reconstructed wavefront and its associated Zernike coefficients. 

The conjugate of this reconstructed wavefront is typically used to generate an actuator command map (in units of actuator stroke length) which is then fed to a Deformable Mirror (DM) to correct for this distortion in real-time.

![System Schematic](./visualizations/reference.jpg)

---

## Interactive Jupyter Notebooks

The calculations, rendering, training, and compilation suites detailed in this project are fully interactive and can be executed via the notebooks located in the `notebook/` folder:

1. **[`kaggle_synthetic_shwfs_generator.ipynb`](./notebook/kaggle_synthetic_shwfs_generator.ipynb):** 
   - Rebuilds the end-to-end WFS pipeline. Renders physical frames, configures system directories, trains the ML reconstructors, and compiles/executes the C POSIX integration test suites.
2. **[`V1_Simulation_TEST.ipynb`](./notebook/V1_Simulation_TEST.ipynb):**
   - The reference execution notebook housing pre-calculated outputs and static telemetry diagrams.
3. **[`Kaggle_RIPRA_WFS_Predictive_AO_Pipeline.ipynb`](./notebook/Kaggle_RIPRA_WFS_Predictive_AO_Pipeline.ipynb):**
   - Implements the deep-learning sequence model pipeline, training **LSTM predictors** for loop lag compensation, turbulence regime classification, and parameter estimation.
4. **[`Kaggle_RIPRA_ML_Pipeline.ipynb`](./notebook/Kaggle_RIPRA_ML_Pipeline.ipynb):**
   - Training pipeline to map centroid displacements to Zernike modal coefficients.
5. **[`Kaggle_RIPRA_ML_Pipeline_baseline.ipynb`](./notebook/Kaggle_RIPRA_ML_Pipeline_baseline.ipynb):**
   - Training pipeline for baseline model configurations.

---

## Wavefront Diagnostics and Telemetry Highlights

Below are the key visual outcomes of the physical simulation and closed-loop control loops. 

### 1. Wavefront Optical Path Difference (OPD) Phase Map
![Wavefront Telemetry](./visualizations/simulation/81_advanced_wavefront_analysis__telemetr.png)
* **Description:** Renders the 2D reconstructed phase screen ($W(x,y)$) alongside a 3D elevation map showing peaks (positive phase delay) and valleys (negative phase delay) of the optical aberration.
* **Impact:** Confirms high-fidelity reconstruction of low-order modes (Tip, Tilt, Defocus) across the circular pupil boundary.

### 2. Deep Learning Reconstructor Accuracy Benchmarks
![ML Dashboard](./visualizations/simulation/111_model_performance_diagnostics_dashbo.png)
* **Description:** Displays MLP vs. CNN training loss convergence, defocus mode regression accuracy, and mode-by-mode Pearson correlation comparison.
* **Impact:** The Conv2D CNN reconstructor achieves a test MSE of **$0.001957$** (mean correlation of **$99.97\%$**), representing a **$4.6\times$** accuracy gain over the MLP baseline.

### 3. Predictive AO Lag Compensation
![Predictive AO](./visualizations/predictive_ao.png)
* **Description:** Trains an LSTM predictor on historical Zernike sequences. Under 1-frame latency, a standard integrator control loop diverges (green curve), whereas the LSTM predictor (blue curve) remains stable, reducing residual RMS error by $6.6\%$.
* **Impact:** Prevents loop instability in high-speed optical systems operating under hardware delay.

---

## Real-Time Processing Performance Benchmarks

The real-time pipeline executes in sub-milliseconds on standard CPU threads, making it fully qualified for high-frequency ($1\text{ kHz}$) closed-loop control:

| Pipeline Phase | Algorithm | Latency ($\mu\text{s}$) |
|---|---|---|
| **Centroiding** | Thresholded Center of Gravity (TCoG) | $482\,\mu\text{s}$ |
| **Reconstruction** | Fried Geometry Zonal Matrix Solver | $194\,\mu\text{s}$ |
| **DM Actuator Mapping** | Influence Coupling Matrix multiplication | $85\,\mu\text{s}$ |
| **Total Latency** | End-to-End Loop | **$761\,\mu\text{s}$** |

---

## Installation and Execution Guide

### 1. Build the POSIX C Library
Compile the static archive `librippra.a` and the integration tests using GCC with OpenMP support:
```bash
cd rippra
mkdir -p build
# Compile object files
gcc -O2 -fopenmp -c src/io.c -o build/io.o -Iinclude
gcc -O2 -fopenmp -c src/la.c -o build/la.o -Iinclude
gcc -O2 -fopenmp -c src/centroid.c -o build/centroid.o -Iinclude
gcc -O2 -fopenmp -c src/recon.c -o build/recon.o -Iinclude
gcc -O2 -fopenmp -c src/rippra_api.c -o build/rippra_api.o -Iinclude

# Link static archive
ar rcs build/librippra.a build/io.o build/la.o build/centroid.o build/recon.o build/rippra_api.o

# Build test suites
gcc -O2 -fopenmp tests/test_full_pipeline.c build/io.o build/la.o build/centroid.o build/recon.o build/rippra_api.o -Iinclude -lm -o build/test_full_pipeline
gcc -O2 -fopenmp tests/test_recon.c build/io.o build/la.o build/centroid.o build/recon.o build/rippra_api.o -Iinclude -lm -o build/test_recon
```

### 2. Run the C Verification Tests
Verify centroiding accuracy, zonal/modal solvers, ground-truth validations, and closed-loop DM convergence:
```bash
./build/test_full_pipeline
./build/test_recon
```

**Representative Output Metrics (Validated against Ground Truth):**
* **Centroiding Accuracy:** `Displacement RMSE: 0.0968 px` (asserted $< 0.25$ px)
* **Reconstruction Accuracy:** `Scaled Zernike RMSE: 0.0154 rad` (asserted $< 0.5$ rad)
* **Strehl Ratio (Marechal):** `1.0000` (flat) / computed dynamically from phase variance
* **DM Correction Residual:** Converges to `< 1e-8 rad` in 6 iterations (gain = 0.5)

### 3. Run the ML Pipeline
Install dependencies and launch the Jupyter Notebook environment:
```bash
pip install torch numpy matplotlib pandas scipy onnx onnxruntime
jupyter notebook
```
Open `notebook/kaggle_synthetic_shwfs_generator.ipynb` to customize parameters, render new calibration frames, or train models.

### 4. Unified Reproducibility Sweep
To verify the entire calibration, dataset generation, MLP model training, and ONNX temporal simulation in a single command, run the unified cross-platform sweep script:
```bash
python rippra/tools/reproduce_all.py
```
This script dynamically builds the C assets, runs C calibration, generates a 500-sample Kolmogorov turbulence dataset, trains an MLP reconstructor for 3 epochs, and runs the ONNX validation and predictive AO sequence checks.

---

## Problem Statement Visualizations

### 1. Example of Wavefront Sensor (WFS) Frame
![WFS Frame](./visualizations/Example%20of%20wavefront%20sensor%20(WFS)%20frame.webp)

### 2. Spot Deviation on Detector due to Distorted Wavefront
![Spot Deviation](./visualizations/Schematic%20showing%20spot%20deviation%20on%20detector%20due%20to%20distorted%20wavefront.webp)
