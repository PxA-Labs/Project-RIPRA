# Wavefront Reconstruction Techniques

Wavefront reconstruction converts the measured discrete slope vector $\mathbf{s} = [\theta_x, \theta_y]^T$ into a 2D phase map $W(x, y)$.

## 1. Zonal Reconstruction (Fried Geometry)
In zonal reconstruction, the phase is solved directly at specific grid points.
We align the lenslet grid and phase/actuator grid in a **Fried Geometry**:
* Phase points (reconstructed nodes) are located at the corners of each sub-aperture.
* Slope measurements (averages across the sub-aperture) are located at the center.

### Fried Matrix System
For a sub-aperture $(i, j)$ defined by corner phase points $W_{i,j}, W_{i+1,j}, W_{i,j+1}, W_{i+1,j+1}$:

$$\theta_x(i, j) \approx \frac{1}{2d} \left[ (W_{i+1, j+1} - W_{i, j+1}) + (W_{i+1, j} - W_{i, j}) \right]$$

$$\theta_y(i, j) \approx \frac{1}{2d} \left[ (W_{i+1, j+1} - W_{i+1, j}) + (W_{i, j+1} - W_{i, j}) \right]$$

This forms a linear matrix system:

$$\mathbf{s} = \mathbf{G} \mathbf{\phi}$$

Where:
* $\mathbf{s}$ is the slope measurement vector.
* $\mathbf{G}$ is the geometry interaction matrix.
* $\mathbf{\phi}$ is the phase vector at the grid points.

The phase is reconstructed using the least-squares pseudo-inverse:

$$\mathbf{\phi} = (\mathbf{G}^T \mathbf{G})^{-1} \mathbf{G}^T \mathbf{s}$$

Singular Value Decomposition (SVD) is used to invert $\mathbf{G}^T \mathbf{G}$ and eliminate the piston (flat phase) singularity.

---

## 2. Modal Reconstruction (Zernike Polynomials)
Modal reconstruction represents the wavefront as a linear combination of continuous orthogonal basis functions (Zernike Polynomials):

$$W(x, y) = \sum_{k=1}^K a_k Z_k(x, y)$$

The slope measurements are fitted to Zernike derivatives:

$$\theta_x(x, y) = \sum_{k=1}^K a_k \frac{\partial Z_k}{\partial x}$$

$$\theta_y(x, y) = \sum_{k=1}^K a_k \frac{\partial Z_k}{\partial y}$$

Which simplifies to:

$$\mathbf{s} = \mathbf{Z}' \mathbf{a} \implies \mathbf{a} = (\mathbf{Z}'^T \mathbf{Z}')^{-1} \mathbf{Z}'^T \mathbf{s}$$
