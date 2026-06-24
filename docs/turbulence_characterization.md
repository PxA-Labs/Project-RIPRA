# Turbulence Characterization

We characterize atmospheric turbulence using statistical parameters derived from the SH-WFS slope variations.

## 1. Fried Parameter ($r_0$)
The Fried parameter (coherence length) $r_0$ describes the spatial scale of wavefront phase fluctuations. 
Assuming Kolmogorov turbulence theory, the variance of the wavefront slopes in a sub-aperture of width $d$ is:

$$\sigma^2_{\theta_x} = 0.170 \left(\frac{\lambda}{d}\right)^2 \left(\frac{d}{r_0}\right)^{5/3} = 0.170 \lambda^2 d^{-1/3} r_0^{-5/3}$$

Solving for $r_0$:

$$r_0 = \left( \frac{0.170 \lambda^2 d^{-1/3}}{\sigma^2_{\theta_x}} \right)^{3/5}$$

By calculating the variance of the $x$ and $y$ slopes across a time-series of frames, we can compute a stable spatial estimate of $r_0$.

---

## 2. Coherence Time ($\tau_0$)
Coherence time $\tau_0$ describes the temporal stability of the turbulence. According to the Taylor frozen-flow hypothesis, the turbulence structure is frozen and simply advected by a horizontal wind velocity $v$:

$$\tau_0 = 0.314 \frac{r_0}{v}$$

### Wind Velocity & Correlation Method
To estimate $v$ or $\tau_0$ directly from time-series slope data, we compute the temporal auto-covariance of the slope coefficients (or Zernike tilt terms):

$$C_s(\Delta t) = \langle s(t) \cdot s(t + \Delta t) \rangle$$

The coherence time $\tau_0$ is proportional to the decay rate of this temporal correlation function (e.g., the time delay at which $C_s(\Delta t)$ falls to $1/e$ of its initial value $C_s(0)$).
