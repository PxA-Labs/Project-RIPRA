# Project RIPRA (ऋप्र)

Developing and optimizing algorithms for **Wavefront Reconstruction** and **Turbulence Characterization** using Shack-Hartmann Wavefront Sensor (SH-WFS) time-series data.

## Description
Turbulence in the atmosphere distorts a plane-parallel wavefront propagating through it. A Shack-Hartmann Wavefront Sensor (SH-WFS) samples this distorted wavefront using an array of small lenslets (Microlens Array - MLA). The MLA creates a spot-field on the camera detector, and the spatial deviation of these spots from their reference positions is used to derive the reconstructed wavefront and its associated Zernike coefficients. 

The conjugate of this reconstructed wavefront is typically used to generate an actuator command map (in units of actuator stroke length) which is then fed to a Deformable Mirror (DM) to correct for this distortion in real-time.

---

## Objectives
* **Process SH-WFS frames** collected during turbulence simulated in the laboratory.
* **Develop fast image processing algorithms** to perform wavefront reconstruction, turbulence characterization, and actuator map determination for the deformable mirror.
* **Optimize for Real-Time Execution**: Since atmospheric turbulence has an inherent coherence timescale of the order of milliseconds, the reconstruction algorithm must be fast enough to measure the distortion and correct for it (< 10 ms).
* **Derive Statistical Parameters**: Characterize the strength and dynamics of turbulence by computing the Fried parameter ($r_0$) and the coherence time ($\tau_0$) from the same time-series data.

---

## Expected Outcomes
1. **Reconstructed Wavefront Phase Maps** ($W(x_i, y_i)$) for each SH-WFS frame.
2. **Turbulence Characterization** in terms of:
   * **Fried parameter** ($r_0$)
   * **Coherence time** ($\tau_0$)
3. **Deformable Mirror Actuator Maps** ($A(x_i, y_i)$) for each reconstructed wavefront map, incorporating:
   * Actuator stroke length mapping.
   * Inter-actuator coupling compensation.

---

## Data Requirements
The dataset to be provided includes:
* **Time-Series of SH-WFS Frames**: Sequence of `.bmp` files captured at short intervals (a few milliseconds) by a science-grade camera.
* **Frame Metadata**: Pixel size and frame resolution.
* **Microlens Array (MLA) Parameters**: Lenslet size (pitch), number of lenslets, and focal length.
* **Pupil Diameter**: Size of the turbulated beam.
* **Deformable Mirror (DM) Parameters**: Actuator grid geometry and inter-actuator coupling characteristics.

---

## Suggested Tools & Technologies
* **Language**: C is advised to achieve the required computational efficiency for real-time applications (corrections at rates faster than 10 ms).
* **Methods**: Zonal/modal reconstruction using orthogonal polynomials (Zernike polynomials) or direct integration methods.
* **Libraries**: Optimization and linear algebra libraries may be utilized to perform complex mathematical computations.

---

## Expected Processing Steps
1. **Centroid Detection**: Identify the centroid position of each spot associated with a sub-aperture in the WFS frames using a robust centroiding algorithm (e.g., thresholded center of gravity).
2. **Displacement Calculation**: For each spot, calculate the deviation from its calibrated reference position.
3. **Wavefront Reconstruction**: Reconstruct the wavefront phase map using zonal/modal techniques. The lenslet grid of the MLA and the actuator grid of the DM are arranged in a **Fried geometry**.
4. **Turbulence Characterization**: Use the reconstructed maps or the Zernike coefficients to derive statistical turbulence metrics.
5. **Actuator Mapping**: Apply the conjugate of the reconstructed wavefront to compute the DM actuator command voltages/strokes, accounting for inter-actuator mechanical coupling.

---

## Evaluation Criteria
* **Accuracy**: Successful reconstruction of wavefront phase maps ($W(x_i, y_i)$) conforming to the turbulence characteristics.
* **Validation**: Correct estimation of the physical statistical parameters of the turbulence.
* **Performance**: Speed and computational efficiency of the algorithms (real-time suitability).
