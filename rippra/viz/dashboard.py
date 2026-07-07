#!/usr/bin/env python3
# viz/dashboard.py - Main RIPRA Visualization Dashboard
# Renders all Phase 7 visualizations and generates a summary HTML page.
import os, sys, webbrowser, base64
from io import BytesIO
try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except AttributeError:
    pass
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wavefront_viz import WavefrontVisualizer
from zernike_dashboard import ZernikeDashboard
from turbulence_dashboard import TurbulenceDashboard
from performance_monitor import PerformanceMonitor
from dm_heatmap import DMHeatmap

class RIPRADashboard:
    def __init__(self):
        self.base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.out_dir = os.path.join(self.base, '..', 'visualizations')
        os.makedirs(self.out_dir, exist_ok=True)

    def render_all(self):
        print("=" * 55)
        print("  RIPRA Visualization Dashboard")
        print("=" * 55)

        WavefrontVisualizer().render_all()
        ZernikeDashboard().render_all()
        TurbulenceDashboard().render_all()
        PerformanceMonitor().render()
        DMHeatmap().plot_dm_heatmap()

        self._generate_html()

        print("\n" + "=" * 55)
        print(f"  All visualizations saved to: {self.out_dir}")
        print("  Open index.html in browser to view the dashboard.")
        print("=" * 55)

    def _generate_html(self):
        images = {
            'Spot Centroid Offsets': 'spot_centroid_offsets.png',
            'Wavefront Phase Map (2D)': 'wavefront_phase_2d.png',
            'Wavefront Profile (3D)': 'wavefront_3d.png',
            'Zernike Coefficients': 'zernike_bar_chart.png',
            'Low-Order Time Series': 'zernike_time_series.png',
            'Turbulence Telemetry': 'turbulence_telemetry.png',
            'Turbulence Regime': 'turbulence_regime.png',
            'Performance Panel': 'performance_panel.png',
            'DM Actuator Heatmap': 'dm_actuator_heatmap.png',
        }

        html = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>RIPRA Wavefront Sensing Dashboard</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { background: #0d0d1a; color: #e0e0e0; font-family: 'Segoe UI', Arial, sans-serif; padding: 20px; }
  h1 { text-align: center; color: #00ff88; font-size: 28px; margin-bottom: 5px; letter-spacing: 2px; }
  h2 { color: #4488ff; font-size: 16px; border-bottom: 2px solid #00ff88; padding-bottom: 8px; margin: 30px 0 15px; text-transform: uppercase; letter-spacing: 1px; }
  .subtitle { text-align: center; color: #888; font-size: 13px; margin-bottom: 30px; }
  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(500px, 1fr)); gap: 20px; }
  .card { background: #1a1a2e; border-radius: 10px; padding: 15px; border: 1px solid #2a2a4a; }
  .card h3 { color: #ccc; font-size: 14px; margin-bottom: 10px; }
  .card img { width: 100%; height: auto; border-radius: 6px; }
  .footer { text-align: center; color: #555; font-size: 12px; margin-top: 40px; padding-top: 20px; border-top: 1px solid #2a2a4a; }
</style>
</head>
<body>
<h1>RIPRA (&#x090b;&#x092a;&#x094d;&#x0930;)</h1>
<p class="subtitle">Adaptive Optics &mdash; Shack-Hartmann Wavefront Reconstruction Dashboard</p>
"""

        sections = [
            ('Wavefront Visualization', [
                'Spot Centroid Offsets', 'Wavefront Phase Map (2D)', 'Wavefront Profile (3D)'
            ]),
            ('Zernike Coefficient Dashboard', [
                'Zernike Coefficients', 'Low-Order Time Series'
            ]),
            ('Turbulence Analytics', [
                'Turbulence Telemetry', 'Turbulence Regime'
            ]),
            ('System Performance', [
                'Performance Panel'
            ]),
        ]

        for section_name, section_images in sections:
            html += f'<h2>{section_name}</h2>\n<div class="grid">\n'
            for name in section_images:
                filename = images[name]
                path = os.path.join(self.out_dir, filename)
                if os.path.exists(path):
                    with open(path, 'rb') as f:
                        data = base64.b64encode(f.read()).decode('utf-8')
                    html += f'  <div class="card"><h3>{name}</h3><img src="data:image/png;base64,{data}" alt="{name}"></div>\n'
            html += '</div>\n'

        html += """<div class="footer">
RIPRA Wavefront Reconstruction &amp; Turbulence Characterization &mdash; Real-Time Adaptive Optics Pipeline
</div>
</body>
</html>"""

        path = os.path.join(self.out_dir, 'index.html')
        with open(path, 'w', encoding='utf-8') as f:
            f.write(html)
        print(f"  Saved: {path}")

def main():
    RIPRADashboard().render_all()

if __name__ == '__main__':
    main()
