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
