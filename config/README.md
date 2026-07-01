# Project RIPRA Configuration System

This directory contains configuration files for the Project RIPRA Adaptive Optics pipeline.

## Configuration Files

The repository maintains two configuration files for different contexts:

1. **`rippra/config/system.conf` (Authoritative for C Engine)**
   * **Location:** `rippra/config/system.conf`
   * **Purpose:** This file is read at runtime by the compiled C binaries and test suites.
   * **Format:** Key-value text format (`key = value`).
   * **Current Settings:**
     * Microlens array: 140 sub-apertures (lenslets), focal length = 18 mm, pitch = 300 µm
     * Camera: Pixel size = 7.4 µm, resolution = 648 x 492
     * Wavelength: 632.8 nm (HeNe laser)
     * DM: 12x12 actuators, coupling = 0.15

2. **`config/default.yaml` (Project Spec / Python Pipeline)**
   * **Location:** `config/default.yaml`
   * **Purpose:** High-level project specifications, synthetic generators, or template configurations for Python scripts.
   * **Format:** YAML format.
   * **Current Settings:**
     * Microlens array: 20x20 sub-apertures (n_lenslets = 20), focal length = 18 mm, pitch = 150 µm
     * Camera: Pixel size = 5.6 µm, resolution = 1024 x 1024
     * Wavelength: 500 nm (Green light)
     * DM: 21 actuators, coupling = 0.15

## Synchronization Warning

> [!WARNING]
> If you make changes to physical parameters such as the wavelength, microlens focal length, or pixel size, ensure that you update **both** `rippra/config/system.conf` and `config/default.yaml` if you are running hybrid C-Python pipelines to prevent calibration discrepancies.
