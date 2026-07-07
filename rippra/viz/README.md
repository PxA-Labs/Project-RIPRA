# Visualization Dashboards — `rippra/viz/`

Python matplotlib + HTML dashboards for real-time pipeline monitoring.

## Files

| File | Purpose |
|---|---|
| `wavefront_viz.py` | 2D phase map, 3D surface, spot centroid offsets |
| `zernike_dashboard.py` | Modal weight bar chart, low-order time-series tracking |
| `turbulence_dashboard.py` | r₀/τ₀ telemetry, D/r₀ regime classification |
| `performance_monitor.py` | 6-panel latency/FPS/CPU/GPU/memory monitor |
| `dashboard.py` | Master dashboard — renders all visualizations + HTML index |
| `pipeline_dashboard.py` | Three-node pipeline dashboard generator |
| `generate_3d_animation.py` | Matplotlib 3D wavefront GIF from real time-series CSV |
| `animate_wavefront.py` | Wavefront animation utilities |
| `dm_heatmap.py` | DM actuator stroke heatmap over actuator grid |

## Dashboard Pages

| Dashboard | File | Description |
|---|---|---|
| Pipeline Dashboard | `pipeline_dashboard.html` | 3-node: SH-WFS input → processing → wavefront |
| Full Dashboard | `index.html` | All 8 plots in single-page dark-themed layout |
| Wavefront 3D | `wavefront_3d.png` | RdBu_r surface with circular pupil |
| Zernike Time Series | `zernike_time_series.png` | 5 low-order modes over 500 frames |
| Turbulence Regime | `turbulence_regime.png` | D/r₀ with Weak/Moderate/Strong zones |
| Performance Panel | `performance_panel.png` | 6-panel system monitor |

## Usage

```bash
# Render all visualizations
python dashboard.py

# Render pipeline dashboard
python pipeline_dashboard.py

# Generate 3D animation GIF (requires time_series.csv)
python generate_3d_animation.py
```

All outputs saved to `visualizations/`.

### DM Heatmap

```bash
python dm_heatmap.py
```

Output: `visualizations/dm_actuator_heatmap.png` — two-panel figure showing the DM actuator commands (left) and the reconstructed zonal phase map (right), with the circular pupil boundary overlaid.

## Pipeline Dashboard

The three-node pipeline dashboard (`pipeline_dashboard.html`) is self-contained (no CDN dependencies):
- **Node 1:** SH-WFS Input — animated 8×8 donut spots (HTML canvas)
- **Node 2:** Processing Pipeline — algorithm list + latency/throughput specs
- **Node 3:** Reconstructed Wavefront — matplotlib-animated 3D GIF with colorbar
