# Deformable Mirror Actuator Mapping

This document describes how we map the conjugate of the reconstructed wavefront to Deformable Mirror (DM) actuator control values.

## 1. Wavefront Conjugation
To correct the distortion, the DM surface must match the negative (conjugate) of the reconstructed wavefront phase $\mathbf{\phi}$:

$$\mathbf{w}_{DM} = -\mathbf{\phi}$$

## 2. Inter-Actuator Coupling
Because Deformable Mirror actuators are physically connected to a continuous membrane or faceplate, pushing one actuator pulls or pushes the neighboring membrane. This is known as **inter-actuator coupling**.

The resulting mirror profile $\mathbf{w}_{DM}$ is related to the applied actuator strokes (commands) $\mathbf{v}$ by:

$$\mathbf{w}_{DM} = \mathbf{C} \mathbf{v}$$

Where $\mathbf{C}$ is the inter-actuator coupling matrix.

### Coupling Matrix Model
A common model for a 2D actuator grid is a nearest-neighbor coupling matrix where the command at actuator $i$ leaks to its neighbor $j$:
* $C_{ii} = 1.0$ (Self-influence)
* $C_{ij} = c$ (Nearest neighbors)
* $C_{ij} = c^2$ (Diagonal neighbors)

Where $c \in [0.10, 0.25]$ is the coupling factor.

### Inversion and Command Calculation
To find the required actuator commands $\mathbf{v}$, we invert the coupling matrix:

$$\mathbf{v} = \mathbf{C}^{-1} \mathbf{w}_{DM} = -\mathbf{C}^{-1} \mathbf{\phi}$$
