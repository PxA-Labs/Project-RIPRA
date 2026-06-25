# Project RIPRA (ऋप्र): Roadmap of Future Phases

This document details the objectives, expected technical architectures, and success criteria for **Phases 6 through 11** of **Project RIPRA (Wavefront Reconstruction & Turbulence Characterization)**.

---

## Phase 6: Real-Time System Development

Adaptive Optics (AO) systems must run in closed-loop configurations to keep pace with changing atmospheric turbulence. For high-fidelity astronomical or satellite communications, the entire sensing-to-correction cycle must have latency $< 10\text{ ms}$ (ideally $< 1.0\text{ ms}$).

```mermaid
graph TD
    A[Raw Frame Capture] -->|Centroid Tracking| B(Spot Displacements)
    B -->|Zonal/Modal Reconstruction| C(Wavefront Phase Map)
    C -->|Actuator Command Mapping| D(DM Commands)
    D -->|Actuator Response| E[Deformable Mirror]
    style B fill:#f9f,stroke:#333,stroke-width:2px
    style C fill:#ccf,stroke:#333,stroke-width:2px
```

### Checkpoint 6.1 – Pipeline Optimization (OpenMP)
* **Goal:** Reduce classical CPU C pipeline latency to $< 1.0\text{ ms}$ per frame.
* **Approach:**
  * **Multithreading:** Inject OpenMP directives (`#pragma omp parallel for`) into high-overhead loops:
    * Spot centroid tracking: Calculate the local Center of Gravity (CoG) for all 127 spots in parallel.
    * Matrix multiplication: Parallelize row multiplications in `rippa_matvec` for geometry and Zernike derivative matrix projections.
    * Modal numerical integration: Multi-thread the circular disk area integration during system startup.
  * **Algorithmic Pruning:** Pre-compute the SVD pseudo-inverses ($\mathbf{G}^+$ and $\mathbf{Z}'^+$) during calibration so that the real-time path only executes fast matrix-vector products.

### Checkpoint 6.2 – GPU Acceleration
* **Goal:** Accelerate both classical centroiding and AI/ML model inference on GPUs.
* **Approach:**
  * **AI/ML GPU Pipeline:** Run the `WavefrontCNN` and sequence LSTMs directly on CUDA execution devices (`device = 'cuda'`).
  * **Classical GPU Paths:** Explore CUDA/OpenCL parallelization for full-frame raw image processing (e.g. thresholding, filtering, and centroid extraction).

### Checkpoint 6.3 – Real-Time Processing Integration
* **Goal:** Create a simulated low-latency streaming pipeline to process time-series frames continuously.
* **Approach:**
  * Implement double-buffering (ping-pong buffers) where one buffer stores incoming camera frames while the other is being processed.
  * Establish a circular queue to handle multi-threaded frame acquisition and processing pipelines.

---

## Phase 7: Visualization & Dashboard

A premium user interface is essential to display the wavefront characteristics, reconstruction accuracy, and deformable mirror states in real-time.

### Checkpoint 7.1 – Wavefront Visualization
* **2D Phase Maps:** Render a 2D color contour plot of the reconstructed phase profile $\phi(x, y)$ over the pupil aperture.
* **3D Wavefront Profiles:** Render interactive 3D surface meshes (using Plotly, Three.js, or Matplotlib) showing the physical shape of the wavefront deviations in microns.
* **Spot Centroid Offsets:** Render the camera frame grid overlaid with reference centroids (green circles) and aberrated centroids (red crosses), with vectors indicating displacement magnitudes.

### Checkpoint 7.2 – Zernike Coefficient Dashboard
* **Modal Weight Distribution:** Render dynamic bar charts displaying the Zernike coefficients $a_2 \dots a_{21}$.
* **Time-Series Tracking:** Provide a scrolling line graph to track the evolution of low-order modes (e.g., Tilt, Defocus, Astigmatism) over time.

### Checkpoint 7.3 – Turbulence Analytics Dashboard
* **Turbulence Parameters:** Display large, premium telemetry readouts for the Fried parameter ($r_0$), Coherence time ($\tau_0$), and estimated wind speed vectors.
* **regime Telemetry:** Display the active classification status of the turbulence (Weak, Moderate, Strong) based on sequential LSTM outputs.

### Checkpoint 7.4 – Loop Performance Monitoring
* **Loop Status:** Indicate closed-loop vs. open-loop states, frame rates (FPS), memory footprint, and CPU/GPU utilization percentages.

---

## Phase 8: Evaluation & Validation

A thorough verification of model performance, limits of correctness, and ablation parameters ensures the system's operational readiness.

### Checkpoint 8.1 – Baseline Comparison
* Create a master benchmark script comparing all implemented algorithms under identical noise conditions:
  $$\text{Classical Zonal vs. Classical Modal vs. WavefrontMLP vs. WavefrontCNN}$$
* Metrics: Reconstruction RMSE, Pearson correlation coefficients, and peak-to-valley wavefront values.

### Checkpoint 8.2 – Noise & Robustness Testing
* Evaluate reconstruction accuracy under varying photon levels (Poisson noise) and thermal readout noise (Gaussian noise).
* **Spot Occlusion / Dropout Test:** Evaluate reconstruction performance when a subset of sub-aperture centroids is blocked (simulating pupil obscuration, spiders, or dead spots in the detector).

### Checkpoint 8.3 – Ablation Study
* Systematically evaluate network design choices:
  * Impact of LSTM lookback window lengths ($L = 5, 10, 20$).
  * Impact of CNN grid resolutions ($13 \times 13$ vs $15 \times 15$).
  * Impact of model architectures (MLP, ResNet, sequential Transformers).

### Checkpoint 8.4 – Performance Benchmarking
* Profile process execution memory footprints, startup compilation latency, per-frame execution averages, and latency jitter profiles (variance in execution times).

---

## Phase 9: Deployment & Packaging

To make the codebase accessible to actual AO systems, it must be compiled, containerized, and packaged with clean programming APIs.

### Checkpoint 9.1 – Model Packaging
* **ONNX Export:** Export trained PyTorch models (CNN, LSTMs) to Open Neural Network Exchange (ONNX) format for fast, hardware-independent runtime execution.
* **Dynamic Libraries:** Package the C modules into dynamic libraries (`librippra.so` on Linux, `rippra.dll` on Windows) for easy embedding.

### Checkpoint 9.2 – API Development
* **Python Bindings:** Create standard bindings using `ctypes` or `CFFI` so that Python scripts can execute the high-performance classical C reconstructor directly.
* **C APIs:** Define clear header interfaces for integrating model runtimes (via ONNX Runtime C API or LibTorch) directly into C++ control loops.

### Checkpoint 9.3 – Deployment Pipeline
* Set up automated compilation scripts (CMake/Make) and container structures (Docker) to compile and run the codebase on target embedded systems (e.g., NVIDIA Jetson, Raspberry Pi, or industrial PCs).

### Checkpoint 9.4 – User Documentation
* Write comprehensive manuals detailing system configuration (`system.conf`), API function signatures, calibration procedures, and ML training instructions.

---

## Phase 10: Final Submission

The culmination of the project involves compiling, formatting, and presenting all research findings, code structures, and demonstrations.

* **Checkpoint 10.1 – GitHub Repository:** Fully clean, refactor, and format all source directories, ensuring comments conform to coding standards and a comprehensive `README.md` is provided at the root.
* **Checkpoint 10.2 – Technical Report:** Author a LaTeX technical report/academic paper detailing the mathematical foundations, implementations, machine learning models, and comparative evaluation results.
* **Checkpoint 10.3 – Demo Video:** Record a high-quality video showing the real-time C reconstruction benchmark running, the ML models training, and the visualization dashboards rendering live wavefront maps.
* **Checkpoint 10.4 – Presentation Deck:** Compile a slideshow covering problem analysis, mathematical frameworks, C achievements, deep learning results, real-time latencies, and future outlooks.

---

## Phase 11: Future Extensions

For advanced systems, these research-grade features pave the way toward space-grade operational deployment:

* **Checkpoint 11.1 – Deformable Mirror Control Integration:** Complete closed-loop integration with hardware DM drivers, mapping wavefront reconstruction phase heights directly to actual actuator driver commands.
* **Checkpoint 11.2 – Predictive Adaptive Optics:** Use the trained sequential LSTM models to feed forward predictive correction shapes to the Deformable Mirror, compensating for the physical lag time of the actuators and sensor integration.
* **Checkpoint 11.3 – Embedded FPGA Deployment:** Implement classical centroiding and reconstruction matrix operations inside FPGA/VHDL modules to achieve sub-microsecond latency.
