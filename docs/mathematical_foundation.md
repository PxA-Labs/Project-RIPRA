# Mathematical Foundation

This document outlines the core mathematical formulations used for Shack-Hartmann Wavefront Sensing.

## 1. Shack-Hartmann Wavefront Sensing Principle
A Shack-Hartmann sensor consists of a microlens array (MLA) and a detector. An incoming wavefront is divided into a grid of sub-apertures. Each lenslet focuses its portion of light onto the detector, forming a spot.

If the wavefront is flat (collimated beam), the spots form a perfect reference grid. A distorted wavefront causes local tilts in each sub-aperture, causing the spots to shift from their reference positions.

## 2. Centroid Calculation
The centroid $(x_c, y_c)$ of the spot in sub-aperture $(i, j)$ is computed using the **Center of Gravity (CoG)** or **Thresholded Center of Gravity (TCoG)**:

$$x_c = \frac{\sum_{(x,y) \in S_{i,j}} x \cdot \max(I(x,y) - I_{th}, 0)}{\sum_{(x,y) \in S_{i,j}} \max(I(x,y) - I_{th}, 0)}$$

$$y_c = \frac{\sum_{(x,y) \in S_{i,j}} y \cdot \max(I(x,y) - I_{th}, 0)}{\sum_{(x,y) \in S_{i,j}} \max(I(x,y) - I_{th}, 0)}$$

Where:
* $S_{i,j}$ is the pixel region for sub-aperture $(i, j)$.
* $I(x,y)$ is the pixel intensity.
* $I_{th}$ is the background noise threshold.

## 3. Wavefront Slope Estimation
The spot displacements $(\Delta x, \Delta y)$ from the reference coordinates $(x_{ref}, y_{ref})$ are:

$$\Delta x = x_c - x_{ref}$$
$$\Delta y = y_c - y_{ref}$$

These displacements are directly proportional to the local wavefront slopes (gradients) $\theta_x$ and $\theta_y$:

$$\theta_x = \frac{\partial W}{\partial x} \approx \frac{\Delta x \cdot p}{f}$$

$$\theta_y = \frac{\partial W}{\partial y} \approx \frac{\Delta y \cdot p}{f}$$

Where:
* $p$ is the physical pixel pitch.
* $f$ is the focal length of the microlens array.

---

## 4. Zernike Polynomials (Noll Indexing)
Modal wavefront sensing models the phase screen $W(\rho, \theta)$ as a linear combination of orthonormal basis functions:

$$Z_j(\rho, \theta) = N_n^m R_n^{|m|}(\rho) \cdot \begin{cases} \cos(|m|\theta) & \text{for } m \ge 0 \\ \sin(|m|\theta) & \text{for } m < 0 \end{cases}$$

Where:
* $j$ is the 1-based Noll index.
* $n$ is the radial order, and $m$ is the azimuthal frequency.
* $N_n^m = \sqrt{n+1}$ (for $m=0$) and $\sqrt{2(n+1)}$ (for $m \ne 0$) is the Zernike normalization factor.
* $R_n^m(\rho)$ is the radial polynomial defined as:
  $$R_n^m(\rho) = \sum_{s=0}^{(n-m)/2} \frac{(-1)^s (n-s)!}{s! \left(\frac{n+m}{2} - s\right)! \left(\frac{n-m}{2} - s\right)!} \rho^{n-2s}$$

---

## 5. Southwell Area Integration Matrix ($Z'$)
To map discrete slopes to continuous Zernike coefficients, we integrate the Zernike derivatives over the circular area of each sub-aperture lenslet $sa_k$:

$$Z'_{k, j} = \frac{R_p}{\pi R_{sa}^2} \iint_{sa_k} \nabla Z_j(x, y) \, dx \, dy$$

Where:
* $R_p$ is the pupil radius.
* $R_{sa}$ is the sub-aperture microlens radius.
* $(x,y)$ are the normalized pupil coordinates $[-1, 1]$.

Solving this integral using a $15 \times 15$ quadrature grid avoids point-wise derivative mapping errors on higher-order modes.

---

## 6. Marechal Strehl Ratio ($S$)
The Strehl ratio ($S$) is a key optical quality metric describing the focal peak intensity reduction due to aberrations. For small wavefront errors, it is computed via the Marechal approximation:

$$S \approx \exp(-\sigma_\phi^2)$$

Where:
* $\sigma_\phi^2 = \text{var}(\phi) = \langle \phi^2 \rangle - \langle \phi \rangle^2$ is the spatial variance of the reconstructed phase screen in radians.
* $S = 1.0$ represents a diffraction-limited, aberration-free optical system.
