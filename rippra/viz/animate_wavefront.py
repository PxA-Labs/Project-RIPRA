import os, sys, math, json, tempfile
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from functools import lru_cache

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, '..', 'visualizations')
os.makedirs(OUT, exist_ok=True)

sys.path.insert(0, os.path.join(BASE, 'ml'))
from evaluate_inference import load_system_config, noll_to_nm

@lru_cache(maxsize=None)
def zernike_coef(n, m, s):
    abs_m = abs(m)
    num = math.factorial(n - s)
    den = math.factorial(s) * math.factorial((n + abs_m)//2 - s) * math.factorial((n - abs_m)//2 - s)
    sign = 1.0 if s % 2 == 0 else -1.0
    return sign * num / den

def build_zernike_basis(nmodes, n_grid=120):
    pupil_r = 1.0
    x = np.linspace(-pupil_r, pupil_r, n_grid)
    y = np.linspace(-pupil_r, pupil_r, n_grid)
    X, Y = np.meshgrid(x, y)
    rho = np.sqrt(X**2 + Y**2) / pupil_r
    mask = rho <= 1.0
    theta = np.arctan2(Y, X)
    basis = np.zeros((nmodes, n_grid, n_grid))
    for idx in range(nmodes):
        j = idx + 2
        n, m = noll_to_nm(j)
        abs_m = abs(m)
        Z = np.zeros_like(X)
        for s in range((n - abs_m) // 2 + 1):
            c = zernike_coef(n, m, s)
            Z += c * (rho ** (n - 2*s))
        norm = np.sqrt(n + 1) if m == 0 else np.sqrt(2 * (n + 1))
        Z_ij = Z * (np.cos(abs_m * theta) if m >= 0 else np.sin(abs_m * theta))
        basis[idx] = Z_ij * norm
        basis[idx, ~mask] = np.nan
    return basis, mask, X, Y

def load_data():
    dataset_path = os.path.join(BASE, 'data_ai', 'dataset.npz')
    config_path = os.path.join(BASE, 'config', 'system.conf')
    spots_csv = os.path.join(BASE, 'results', 'reference_centroids_c.csv')
    cfg = load_system_config(config_path)
    spots_df = pd.read_csv(spots_csv)
    data = np.load(dataset_path)
    coeffs_all = data['coefficients']
    D_r0 = data['D_r0']
    n_total = len(coeffs_all)
    step = max(1, n_total // 100)
    idx = np.arange(0, n_total, step)[:100]
    return coeffs_all[idx], D_r0[idx], cfg, spots_df

def compute_frames(coeffs_all, basis):
    return np.tensordot(coeffs_all, basis, axes=([1], [0]))

def animate_gif(coeffs_all, D_r0):
    n_frames = len(coeffs_all)
    nmodes = coeffs_all.shape[1]
    n_grid = 80

    basis, mask, X, Y = build_zernike_basis(nmodes, n_grid)
    frames = compute_frames(coeffs_all, basis)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor='#0d0d1a')
    ax1, ax2 = axes
    for ax in axes:
        ax.set_facecolor('#0d0d1a')
        ax.tick_params(colors='white')

    fig.suptitle('Reconstructed Wavefront', color='#00ff88', fontsize=14, fontweight='bold')

    def update(frame):
        Z = frames[frame]
        Z[~mask] = np.nan
        vmax = max(abs(np.nanmin(Z)), abs(np.nanmax(Z))) or 1.0

        ax1.clear()
        ax1.set_facecolor('#0d0d1a')
        ax1.imshow(Z, cmap='RdBu_r', extent=[-2000, 2000, -2000, 2000],
                   vmin=-vmax, vmax=vmax)
        ax1.set_title(f'2D Phase Map (Frame {frame})', color='white')
        ax1.set_xlabel('X (μm)', color='white')
        ax1.set_ylabel('Y (μm)', color='white')
        ax1.tick_params(colors='white')

        ax2.clear()
        ax2.set_facecolor('#0d0d1a')
        X_m = np.ma.masked_where(~mask, X * 2000)
        Y_m = np.ma.masked_where(~mask, Y * 2000)
        Z_m = np.ma.masked_where(~mask, Z)
        ax2.contourf(X_m, Y_m, Z_m, levels=20, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        ax2.set_title(f'Contour (Frame {frame})', color='white')
        ax2.set_xlabel('X (μm)', color='white')
        ax2.set_ylabel('Y (μm)', color='white')
        ax2.tick_params(colors='white')
        ax2.set_aspect('equal')

        dr0 = D_r0[frame]
        regime = 'Weak' if dr0 < 3 else 'Moderate' if dr0 < 6 else 'Strong'
        rms = float(np.nanstd(Z))
        fig.suptitle(f'Frame {frame+1}/{n_frames}  |  D/r₀={dr0:.2f} ({regime}) | WF RMS={rms:.4f} rad',
                     color='#00ff88', fontsize=12, fontweight='bold')

    from matplotlib.animation import FuncAnimation, PillowWriter
    anim = FuncAnimation(fig, update, frames=n_frames, interval=50)
    gif_path = os.path.join(OUT, 'wavefront_animation.gif')
    anim.save(gif_path, writer=PillowWriter(fps=15))
    plt.close(fig)
    print(f"  Saved GIF: wavefront_animation.gif ({n_frames} frames)")

def generate_html(coeffs_all, D_r0):
    n_frames = len(coeffs_all)
    nmodes = coeffs_all.shape[1]

    zernike_coeffs_json = json.dumps(coeffs_all.tolist())
    dr0_json = json.dumps(D_r0.tolist())

    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>RIPRA Animated Wavefront Dashboard</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ background: #0d0d1a; color: #e0e0e0; font-family: 'Segoe UI', Arial, sans-serif; padding: 20px; }}
h1 {{ text-align: center; color: #00ff88; font-size: 26px; margin-bottom: 5px; letter-spacing: 2px; }}
.subtitle {{ text-align: center; color: #888; font-size: 13px; margin-bottom: 20px; }}
.container {{ max-width: 1200px; margin: 0 auto; }}
.canvas-row {{ display: flex; gap: 15px; margin-bottom: 15px; flex-wrap: wrap; justify-content: center; }}
canvas {{ background: #1a1a2e; border-radius: 8px; border: 1px solid #2a2a4a; }}
.controls {{ display: flex; align-items: center; justify-content: center; gap: 12px; margin: 15px 0; flex-wrap: wrap; }}
.controls button {{ background: #1a1a2e; color: #00ff88; border: 1px solid #00ff88; padding: 8px 18px; border-radius: 5px; cursor: pointer; font-size: 14px; }}
.controls button:hover {{ background: #00ff88; color: #0d0d1a; }}
.controls input[type=range] {{ width: 300px; }}
.controls label {{ color: #888; font-size: 13px; }}
.info {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; margin: 15px 0; }}
.info-card {{ background: #1a1a2e; border-radius: 8px; padding: 12px; border: 1px solid #2a2a4a; text-align: center; }}
.info-card .val {{ color: #00ff88; font-size: 20px; font-weight: bold; }}
.info-card .lbl {{ color: #888; font-size: 11px; text-transform: uppercase; margin-top: 4px; }}
</style>
</head>
<body>
<h1>RIPRA Animated Wavefront Dashboard</h1>
<p class="subtitle">Real-time wavefront reconstruction from 50,000 SH-WFS frames (sampled {n_frames})</p>
<div class="container">
<div class="canvas-row">
<canvas id="canvas-2d" width="580" height="540"></canvas>
<canvas id="canvas-ts" width="580" height="540"></canvas>
</div>
<div class="controls">
<button id="btn-play">Play</button>
<button id="btn-prev">&#9664;&#9664;</button>
<input type="range" id="frame-slider" min="0" max="{n_frames-1}" value="0" step="1">
<button id="btn-next">&#9654;&#9654;</button>
<label>Speed: <select id="speed-select">
<option value="80">Slow</option><option value="50" selected>Normal</option><option value="25">Fast</option>
</select></label>
</div>
<div class="info" id="info-panel"></div>
</div>
<script>
const COEFFS = {zernike_coeffs_json};
const D_R0 = {dr0_json};
const N_FRAMES = {n_frames};
const N_MODES = {nmodes};
const N_GRID = 80;

const factCache = [1];
function fact(n) {{ while (factCache.length <= n) factCache.push(factCache.length * factCache[factCache.length - 1]); return factCache[n]; }}

const ZFUNCS = [];
for (let k = 0; k < N_MODES; k++) {{
    let j = k + 2, n = 0, m = 0, cj = 2;
    for (let rn = 1; rn < 20; rn++) {{
        for (let rm = rn % 2; rm <= rn; rm += 2) {{
            if (rm === 0) {{ if (cj === j) {{ n = rn; m = 0; }} cj++; }}
            else {{
                if (cj % 2 === 1) {{ if (cj === j) {{ n = rn; m = -rm; }} if (cj + 1 === j) {{ n = rn; m = rm; }} }}
                else {{ if (cj === j) {{ n = rn; m = rm; }} if (cj + 1 === j) {{ n = rn; m = -rm; }} }}
                cj += 2;
            }}
        }}
    }}
    const am = Math.abs(m), norm = m === 0 ? Math.sqrt(n + 1) : Math.sqrt(2 * (n + 1)), useCos = m >= 0;
    const terms = [];
    for (let s = 0; s <= (n - am) / 2; s++) {{
        const c = ((-1)**s * fact(n - s)) / (fact(s) * fact((n + am)/2 - s) * fact((n - am)/2 - s));
        terms.push({{c, e: n - 2*s}});
    }}
    ZFUNCS.push({{terms, norm, am, useCos}});
}}

function evalWaveform(coeffs) {{
    const N = N_GRID;
    let grid = [], vmax = 0;
    for (let i = 0; i < N; i++) {{
        let row = [];
        for (let j = 0; j < N; j++) {{
            const x = -1 + 2 * j / (N - 1), y = -1 + 2 * i / (N - 1);
            const rho = Math.sqrt(x*x + y*y);
            if (rho > 1) {{ row.push(NaN); continue; }}
            const theta = Math.atan2(y, x);
            let val = 0;
            for (let k = 0; k < N_MODES; k++) {{
                const f = ZFUNCS[k]; let R = 0;
                for (const t of f.terms) R += t.c * Math.pow(rho, t.e);
                val += coeffs[k] * R * f.norm * (f.useCos ? Math.cos(f.am * theta) : Math.sin(f.am * theta));
            }}
            if (Math.abs(val) > vmax) vmax = Math.abs(val);
            row.push(val);
        }}
        grid.push(row);
    }}
    return {{grid, vmax: vmax || 1}};
}}

let cf = 0, playing = false, timer = null;
const c2d = document.getElementById('canvas-2d'), ctx = c2d.getContext('2d');
const cts = document.getElementById('canvas-ts'), tctx = cts.getContext('2d');
const slider = document.getElementById('frame-slider');
const infoPanel = document.getElementById('info-panel');

function heatColor(v, vm) {{
    const t = v / vm;
    if (t > 0) return `rgb(${{Math.round(255*t)}},${{Math.max(0,Math.round(255*(1-t)-50))}},51)`;
    return `rgb(${{Math.round(100*Math.abs(t))}},51,${{Math.max(50,Math.round(255*Math.abs(t)))}})`;
}}

function drawWF({{grid, vmax}}) {{
    const N = grid.length, W = c2d.width, H = c2d.height;
    ctx.fillStyle = '#1a1a2e'; ctx.fillRect(0, 0, W, H);
    const cx = W/2, cy = H/2, r = Math.min(W, H)/2 - 35;
    for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) {{
        const v = grid[i][j]; if (isNaN(v)) continue;
        const x = cx + (2*j/(N-1)-1)*r, y = cy - (2*i/(N-1)-1)*r;
        ctx.fillStyle = heatColor(v, vmax);
        ctx.fillRect(x-1.5, y-1.5, 3, 3);
    }}
    ctx.strokeStyle = '#2a2a4a'; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.arc(cx, cy, r, 0, 2*Math.PI); ctx.stroke();
    ctx.fillStyle = '#00ff88'; ctx.font = 'bold 13px Arial'; ctx.textAlign = 'center';
    ctx.fillText(`Frame ${{cf+1}}/${{N_FRAMES}}`, cx, 22);
    ctx.fillStyle = '#888'; ctx.font = '11px Arial';
    ctx.fillText(`vmax = ${{vmax.toFixed(3)}} rad`, cx, H-12);
}}

function drawTS() {{
    const W = cts.width, H = cts.height, pad = 45;
    tctx.fillStyle = '#1a1a2e'; tctx.fillRect(0, 0, W, H);
    const pw = W - 2*pad, ph = H - 60, y0 = H/2 + 10;
    tctx.strokeStyle = '#2a2a4a'; tctx.lineWidth = 1;
    tctx.beginPath(); tctx.moveTo(pad, y0); tctx.lineTo(W-pad, y0); tctx.stroke();
    tctx.fillStyle = '#888'; tctx.font = '11px Arial'; tctx.textAlign = 'center';
    tctx.fillText('Tip (green) / Tilt (red) / Defocus (blue)', W/2, H-8);
    const series = [[], [], []];
    for (let i = 0; i <= cf; i++) {{ series[0].push(COEFFS[i][0]); series[1].push(COEFFS[i][1]); series[2].push(COEFFS[i][2]); }}
    const cols = ['#00ff88', '#ff4444', '#4488ff'];
    for (let s = 0; s < 3; s++) {{
        tctx.strokeStyle = cols[s]; tctx.lineWidth = 1.5; tctx.beginPath();
        for (let i = 0; i < series[s].length; i++) {{
            const x = pad + (i/(N_FRAMES-1))*pw, y = y0 - (series[s][i]/2.5)*ph/2;
            i === 0 ? tctx.moveTo(x, y) : tctx.lineTo(x, y);
        }} tctx.stroke();
    }}
    const lx = pad + (cf/(N_FRAMES-1))*pw;
    tctx.strokeStyle = '#00ff88'; tctx.lineWidth = 1; tctx.globalAlpha = 0.5;
    tctx.beginPath(); tctx.moveTo(lx, 10); tctx.lineTo(lx, H-30); tctx.stroke();
    tctx.globalAlpha = 1;
}}

function updateInfo() {{
    const dr0 = D_R0[cf];
    const regime = dr0 < 3 ? 'Weak' : dr0 < 6 ? 'Moderate' : 'Strong';
    const {{grid, vmax}} = evalWaveform(COEFFS[cf]);
    let sq = 0, cnt = 0, mn = Infinity, mx = -Infinity;
    for (const row of grid) for (const v of row) {{ if (!isNaN(v)) {{ sq += v*v; cnt++; if (v < mn) mn = v; if (v > mx) mx = v; }} }}
    const rms = Math.sqrt(sq/cnt);
    infoPanel.innerHTML = `
        <div class="info-card"><div class="val">${{cf}}</div><div class="lbl">Frame</div></div>
        <div class="info-card"><div class="val" style="color:#ffaa00">${{dr0.toFixed(2)}}</div><div class="lbl">D/r₀</div></div>
        <div class="info-card"><div class="val" style="color:#ffaa00">${{regime}}</div><div class="lbl">Regime</div></div>
        <div class="info-card"><div class="val" style="color:#4488ff">${{rms.toFixed(4)}}</div><div class="lbl">WF RMS (rad)</div></div>
        <div class="info-card"><div class="val" style="color:#4488ff">${{(mx-mn).toFixed(4)}}</div><div class="lbl">P-V (rad)</div></div>`;
}}

function render() {{ const r = evalWaveform(COEFFS[cf]); drawWF(r); drawTS(); updateInfo(); }}

function play() {{ playing = true; document.getElementById('btn-play').textContent = 'Stop';
    timer = setInterval(() => {{ cf = (cf + 1) % N_FRAMES; slider.value = cf; render(); }}, parseInt(document.getElementById('speed-select').value)); }}
function stop() {{ playing = false; document.getElementById('btn-play').textContent = 'Play'; if (timer) {{ clearInterval(timer); timer = null; }} }}
document.getElementById('btn-play').onclick = () => playing ? stop() : play();
document.getElementById('btn-prev').onclick = () => {{ cf = Math.max(0, cf-1); slider.value = cf; if (playing) stop(); render(); }};
document.getElementById('btn-next').onclick = () => {{ cf = Math.min(N_FRAMES-1, cf+1); slider.value = cf; if (playing) stop(); render(); }};
slider.oninput = function() {{ cf = parseInt(this.value); if (playing) stop(); render(); }};
render();
</script>
</body>
</html>'''
    html_path = os.path.join(OUT, 'animated_wavefront.html')
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"  Saved HTML: animated_wavefront.html ({os.path.getsize(html_path)/1024:.0f} KB)")

def generate_html_3d(coeffs_all, D_r0):
    n_frames = len(coeffs_all)
    nmodes = coeffs_all.shape[1]
    zernike_coeffs_json = json.dumps(coeffs_all.tolist())
    dr0_json = json.dumps(D_r0.tolist())

    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>RIPRA 3D Wavefront Viewer</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ background: #0d0d1a; color: #e0e0e0; font-family: 'Segoe UI', Arial, sans-serif; overflow: hidden; }}
#info {{ position: absolute; top: 15px; left: 50%; transform: translateX(-50%); z-index: 10;
  background: rgba(13,13,26,0.85); padding: 10px 25px; border-radius: 8px; border: 1px solid #00ff88;
  text-align: center; pointer-events: none; }}
#info h1 {{ color: #00ff88; font-size: 18px; letter-spacing: 1px; }}
#info p {{ color: #888; font-size: 12px; margin-top: 2px; }}
#controls {{ position: absolute; bottom: 30px; left: 50%; transform: translateX(-50%); z-index: 10;
  display: flex; align-items: center; gap: 12px; background: rgba(13,13,26,0.85);
  padding: 12px 20px; border-radius: 8px; border: 1px solid #2a2a4a; }}
#controls button {{ background: #1a1a2e; color: #00ff88; border: 1px solid #00ff88; padding: 6px 16px;
  border-radius: 5px; cursor: pointer; font-size: 13px; }}
#controls button:hover {{ background: #00ff88; color: #0d0d1a; }}
#controls input[type=range] {{ width: 280px; }}
#controls label {{ color: #888; font-size: 12px; }}
#stats {{ position: absolute; top: 80px; right: 20px; z-index: 10;
  background: rgba(13,13,26,0.85); padding: 12px 18px; border-radius: 8px; border: 1px solid #2a2a4a;
  font-size: 13px; line-height: 1.8; pointer-events: none; min-width: 150px; }}
#stats .val {{ color: #00ff88; font-weight: bold; }}
#stats .lbl {{ color: #888; }}
</style>
</head>
<body>
<div id="info"><h1>RIPRA 3D Wavefront Viewer</h1><p>Interactive WebGL surface — drag to rotate, scroll to zoom</p></div>
<div id="stats">
  <div><span class="lbl">Frame:</span> <span class="val" id="s-frame">0</span> / <span id="s-total">{n_frames}</span></div>
  <div><span class="lbl">D/r₀:</span> <span class="val" id="s-dr0" style="color:#ffaa00">0.00</span></div>
  <div><span class="lbl">Regime:</span> <span class="val" id="s-regime" style="color:#ffaa00">—</span></div>
  <div><span class="lbl">RMS:</span> <span class="val" id="s-rms" style="color:#4488ff">0.0000</span> rad</div>
  <div><span class="lbl">P-V:</span> <span class="val" id="s-pv" style="color:#4488ff">0.0000</span> rad</div>
</div>
<div id="controls">
  <button id="btn-play">Play</button>
  <button id="btn-prev">&#9664;&#9664;</button>
  <input type="range" id="frame-slider" min="0" max="{n_frames-1}" value="0" step="1">
  <button id="btn-next">&#9654;&#9654;</button>
  <label>Speed: <select id="speed-select">
    <option value="100">Slow</option><option value="50" selected>Normal</option><option value="20">Fast</option>
  </select></label>
  <label><input type="checkbox" id="chk-rotate" checked> Auto-rotate</label>
</div>

<script type="importmap">{{"imports":{{"three":"https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js","three/addons/":"https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/"}}}}</script>
<script type="module">
import * as THREE from 'three';
import {{ OrbitControls }} from 'three/addons/controls/OrbitControls.js';

const COEFFS = {zernike_coeffs_json};
const D_R0 = {dr0_json};
const N_FRAMES = {n_frames};
const N_MODES = {nmodes};
const N_GRID = 60;

const factCache = [1];
function fact(n) {{ while (factCache.length <= n) factCache.push(factCache.length * factCache[factCache.length - 1]); return factCache[n]; }}

const ZFUNCS = [];
for (let k = 0; k < N_MODES; k++) {{
  let j = k + 2, n = 0, m = 0, cj = 2;
  for (let rn = 1; rn < 20; rn++) {{
    for (let rm = rn % 2; rm <= rn; rm += 2) {{
      if (rm === 0) {{ if (cj === j) {{ n = rn; m = 0; }} cj++; }}
      else {{
        if (cj % 2 === 1) {{ if (cj === j) {{ n = rn; m = -rm; }} if (cj + 1 === j) {{ n = rn; m = rm; }} }}
        else {{ if (cj === j) {{ n = rn; m = rm; }} if (cj + 1 === j) {{ n = rn; m = -rm; }} }}
        cj += 2;
      }}
    }}
  }}
  const am = Math.abs(m), norm = m === 0 ? Math.sqrt(n + 1) : Math.sqrt(2 * (n + 1)), useCos = m >= 0;
  const terms = [];
  for (let s = 0; s <= (n - am) / 2; s++) {{
    const c = ((-1)**s * fact(n - s)) / (fact(s) * fact((n + am)/2 - s) * fact((n - am)/2 - s));
    terms.push({{c, e: n - 2*s}});
  }}
  ZFUNCS.push({{terms, norm, am, useCos}});
}}

function buildSurface(coeffs) {{
  const N = N_GRID, scale = 2.0;
  const pos = [], clr = [], idx = [];
  const vals = Array.from({{length: N}}, () => []);
  let vmax = 0;
  for (let i = 0; i < N; i++) {{
    for (let j = 0; j < N; j++) {{
      const x = -1 + 2*j/(N-1), y = -1 + 2*i/(N-1);
      const rho = Math.sqrt(x*x + y*y);
      if (rho > 1) {{ vals[i][j] = NaN; continue; }}
      const theta = Math.atan2(y, x);
      let val = 0;
      for (let k = 0; k < N_MODES; k++) {{
        const f = ZFUNCS[k]; let R = 0;
        for (const t of f.terms) R += t.c * Math.pow(rho, t.e);
        val += coeffs[k] * R * f.norm * (f.useCos ? Math.cos(f.am * theta) : Math.sin(f.am * theta));
      }}
      if (Math.abs(val) > vmax) vmax = Math.abs(val);
      vals[i][j] = val;
    }}
  }}
  vmax = vmax || 1;
  let vertexIdx = 0;
  const idxMap = Array.from({{length: N}}, () => []);
  for (let i = 0; i < N; i++) {{
    for (let j = 0; j < N; j++) {{
      const v = vals[i][j];
      if (isNaN(v)) {{ idxMap[i][j] = -1; continue; }}
      idxMap[i][j] = vertexIdx++;
      const px = (-1 + 2*j/(N-1)) * scale;
      const py = (-1 + 2*i/(N-1)) * scale;
      pos.push(px, v, py);
      const t = v / vmax;
      let r, g, b;
      if (t > 0) {{ r = t; g = Math.max(0, 1 - t*1.5); b = Math.max(0, 0.2 - t*0.2); }}
      else {{ r = Math.max(0, 0.2 + t*0.3); g = Math.max(0, 0.3 + t*0.3); b = 0.6 - t*0.4; }}
      clr.push(r, g, b);
    }}
  }}
  for (let i = 0; i < N-1; i++) {{
    for (let j = 0; j < N-1; j++) {{
      const a = idxMap[i][j], b = idxMap[i][j+1], c = idxMap[i+1][j], d = idxMap[i+1][j+1];
      if (a < 0 || b < 0 || c < 0 || d < 0) continue;
      idx.push(a, b, c, b, d, c);
    }}
  }}
  const geom = new THREE.BufferGeometry();
  geom.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  geom.setAttribute('color', new THREE.Float32BufferAttribute(clr, 3));
  geom.setIndex(idx);
  geom.computeVertexNormals();
  const mat = new THREE.MeshPhongMaterial({{vertexColors: true, side: THREE.DoubleSide,
    specular: 0x222244, shininess: 25, transparent: true, opacity: 0.92,
    flatShading: false}});
  return {{mesh: new THREE.Mesh(geom, mat), vmax}};
}}

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0d0d1a);
const camera = new THREE.PerspectiveCamera(35, innerWidth / innerHeight, 0.1, 100);
camera.position.set(4.5, 3.5, 4.5);
const renderer = new THREE.WebGLRenderer({{antialias: true}});
renderer.setSize(innerWidth, innerHeight);
renderer.setPixelRatio(devicePixelRatio);
document.body.prepend(renderer.domElement);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true; controls.dampingFactor = 0.08;
controls.target.set(0, 0.2, 0); controls.autoRotate = true; controls.autoRotateSpeed = 1.5;
const amb = new THREE.AmbientLight(0x404060); scene.add(amb);
const dl = new THREE.DirectionalLight(0xffffff, 1.0); dl.position.set(5, 10, 5); scene.add(dl);
const fl = new THREE.DirectionalLight(0x4488ff, 0.4); fl.position.set(-5, 0, 5); scene.add(fl);
const gh = new THREE.GridHelper(5, 10, 0x00ff88, 0x2a2a4a); gh.position.y = -0.8; scene.add(gh);

let mesh = null;

function updateFrame() {{
  if (mesh) scene.remove(mesh);
  const result = buildSurface(COEFFS[cf]);
  mesh = result.mesh; scene.add(mesh);
  const dr0 = D_R0[cf];
  const regime = dr0 < 3 ? 'Weak' : dr0 < 6 ? 'Moderate' : 'Strong';
  let sq = 0, cnt = 0, mn = Infinity, mx = -Infinity;
  const p = mesh.geometry.attributes.position.array;
  for (let i = 0; i < p.length; i += 3) {{ const y = p[i+1]; if (!isNaN(y)) {{ sq += y*y; cnt++; if (y < mn) mn = y; if (y > mx) mx = y; }} }}
  document.getElementById('s-frame').textContent = cf;
  document.getElementById('s-dr0').textContent = dr0.toFixed(2);
  document.getElementById('s-regime').textContent = regime;
  document.getElementById('s-rms').textContent = Math.sqrt(sq/cnt).toFixed(4);
  document.getElementById('s-pv').textContent = (mx - mn).toFixed(4);
}}

let cf = 0, playing = false, timer = null;
updateFrame();
function render() {{ controls.update(); renderer.render(scene, camera); requestAnimationFrame(render); }}
render();

document.getElementById('chk-rotate').onchange = function() {{ controls.autoRotate = this.checked; }};
document.getElementById('btn-play').onclick = () => {{
  if (playing) {{ playing = false; document.getElementById('btn-play').textContent = 'Play'; clearInterval(timer); }}
  else {{ playing = true; document.getElementById('btn-play').textContent = 'Stop';
    timer = setInterval(() => {{ cf = (cf+1)%N_FRAMES; document.getElementById('frame-slider').value = cf; updateFrame(); }}, parseInt(document.getElementById('speed-select').value)); }}
}};
document.getElementById('btn-prev').onclick = () => {{ cf = Math.max(0, cf-1); document.getElementById('frame-slider').value = cf; if (playing) {{ clearInterval(timer); playing = false; document.getElementById('btn-play').textContent = 'Play'; }} updateFrame(); }};
document.getElementById('btn-next').onclick = () => {{ cf = Math.min(N_FRAMES-1, cf+1); document.getElementById('frame-slider').value = cf; if (playing) {{ clearInterval(timer); playing = false; document.getElementById('btn-play').textContent = 'Play'; }} updateFrame(); }};
document.getElementById('frame-slider').oninput = function() {{ cf = parseInt(this.value); if (playing) {{ clearInterval(timer); playing = false; document.getElementById('btn-play').textContent = 'Play'; }} updateFrame(); }};
window.onresize = () => {{ camera.aspect = innerWidth / innerHeight; camera.updateProjectionMatrix(); renderer.setSize(innerWidth, innerHeight); }};
</script>
</body>
</html>'''
    html_path = os.path.join(OUT, 'wavefront_3d_viewer.html')
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"  Saved 3D HTML: wavefront_3d_viewer.html ({os.path.getsize(html_path)/1024:.0f} KB)")

def main():
    print("Loading 50,000 frames from dataset.npz (sampling 100 frames)...")
    coeffs_all, D_r0, _, _ = load_data()
    print(f"  Sampled: {coeffs_all.shape[0]} frames, {coeffs_all.shape[1]} Zernike modes")

    print("\n[1/3] Generating animated GIF (this may take a minute)...")
    animate_gif(coeffs_all, D_r0)

    print("\n[2/3] Generating interactive 2D HTML/JS dashboard...")
    generate_html(coeffs_all, D_r0)

    print("\n[3/3] Generating interactive 3D WebGL viewer...")
    generate_html_3d(coeffs_all, D_r0)

    print("\nDone! Files:")
    print("  GIF:        visualizations/wavefront_animation.gif")
    print("  HTML 2D:    visualizations/animated_wavefront.html")
    print("  HTML 3D:    visualizations/wavefront_3d_viewer.html")
    print("\nLive plot:   python rippra/viz/animate_wavefront.py --live")
    print("Live 3D:     python rippra/viz/animate_wavefront.py --live-3d")

def run_live():
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from mpl_toolkits.mplot3d import Axes3D
    plt.ion()
    coeffs_all, D_r0, _, _ = load_data()
    n_frames = len(coeffs_all)
    nmodes = coeffs_all.shape[1]
    n_grid = 40
    basis, mask, X, Y = build_zernike_basis(nmodes, n_grid)
    frames = compute_frames(coeffs_all, basis)
    X_um = X * 2000
    Y_um = Y * 2000
    fig = plt.figure(figsize=(14, 7), facecolor='#0d0d1a')
    ax = fig.add_subplot(111, projection='3d', facecolor='#0d0d1a')
    fig.suptitle('Live 3D Wavefront Reconstruction', color='#00ff88', fontsize=14, fontweight='bold')
    surf = None
    for f in range(n_frames):
        Z = frames[f].copy()
        Z[~mask] = np.nan
        vmax = max(abs(np.nanmin(Z)), abs(np.nanmax(Z))) or 1.0
        ax.clear()
        ax.set_facecolor('#0d0d1a')
        X_m = np.ma.masked_where(~mask, X_um)
        Y_m = np.ma.masked_where(~mask, Y_um)
        Z_m = np.ma.masked_where(~mask, Z)
        surf = ax.plot_surface(X_m, Y_m, Z_m, cmap='gist_rainbow', linewidth=0,
                               antialiased=True, alpha=0.95, vmin=-vmax, vmax=vmax)
        dr0 = D_r0[f]
        regime = 'Weak' if dr0 < 3 else 'Moderate' if dr0 < 6 else 'Strong'
        ax.set_title(f'Frame {f+1}/{n_frames}  |  D/r₀={dr0:.2f} ({regime})  |  RMS={float(np.nanstd(Z)):.4f} rad',
                     color='white', fontsize=11)
        ax.set_xlabel('X (μm)', color='white')
        ax.set_ylabel('Y (μm)', color='white')
        ax.set_zlabel('Phase (rad)', color='white')
        ax.tick_params(colors='white')
        ax.xaxis.pane.set_facecolor('#0d0d1a')
        ax.yaxis.pane.set_facecolor('#0d0d1a')
        ax.zaxis.pane.set_facecolor('#0d0d1a')
        ax.zaxis.pane.set_edgecolor('#2a2a4a')
        ax.grid(True, alpha=0.15, color='gray')
        plt.pause(0.08)
    plt.ioff()
    plt.show()

if __name__ == '__main__':
    if '--live' in sys.argv or '--live-3d' in sys.argv:
        run_live()
    else:
        main()
