import os, sys, math, json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'ml'))
from evaluate_inference import load_system_config, noll_to_nm

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, '..', 'visualizations')
os.makedirs(OUT, exist_ok=True)

def load_data():
    dataset_path = os.path.join(BASE, 'data_ai', 'dataset.npz')
    data = np.load(dataset_path)
    coeffs_all = data['coefficients']
    D_r0 = data['D_r0']
    n_total = len(coeffs_all)
    step = max(1, n_total // 100)
    idx = np.arange(0, n_total, step)[:100]
    return coeffs_all[idx], D_r0[idx]

def generate():
    print("Loading data...")
    coeffs_all, D_r0 = load_data()
    n_frames, nmodes = coeffs_all.shape
    coeffs_json = json.dumps(coeffs_all.tolist())
    dr0_json = json.dumps(D_r0.tolist())

    html = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>RIPRA Adaptive Optics Pipeline</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { background: #0a0a16; color: #e0e0e0; font-family: 'Segoe UI', Arial, sans-serif; min-height: 100vh; display: flex; flex-direction: column; align-items: center; }
.header { text-align: center; padding: 24px 20px 8px; width: 100%; }
.header h1 { color: #00ff88; font-size: 26px; letter-spacing: 3px; font-weight: 300; }
.header h1 span { font-weight: 700; }
.header p { color: #666; font-size: 12px; margin-top: 4px; letter-spacing: 1px; }
.pipeline { display: flex; align-items: center; justify-content: center; gap: 0; padding: 14px 20px 20px; max-width: 1400px; width: 100%; }
.stage { background: linear-gradient(160deg, #141428, #0e0e1e); border-radius: 14px; border: 1px solid #2a2a4a; padding: 18px; flex: 1; min-width: 260px; max-width: 440px; position: relative; overflow: hidden; }
.stage::before { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px; background: linear-gradient(90deg, transparent, #00ff88, transparent); opacity: 0.4; }
.stage-title { color: #00ff88; font-size: 14px; font-weight: 700; text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 12px; text-align: center; }
.arrow-wrap { display: flex; align-items: center; justify-content: center; width: 50px; flex-shrink: 0; }
.arrow-wrap svg { width: 30px; height: 30px; }
#spot-canvas { width: 100%; aspect-ratio: 1; background: #060612; border-radius: 8px; border: 1px solid #1a1a3a; display: block; }
.metrics { margin-top: 10px; }
.metric { display: flex; justify-content: space-between; padding: 6px 0; border-bottom: 1px solid #1a1a30; font-size: 13px; }
.metric:last-child { border-bottom: none; }
.metric .label { color: #999; }
.metric .value { color: #00ff88; font-weight: 600; }
.metric .value.amber { color: #ffaa00; }
.algo-list { list-style: none; font-size: 12.5px; color: #ccc; margin: 6px 0 10px; }
.algo-list li { padding: 4px 0 4px 18px; position: relative; border-bottom: 1px solid #111128; }
.algo-list li:last-child { border-bottom: none; }
.algo-list li::before { content: '\25B8'; position: absolute; left: 0; color: #00ff88; }
#three-container { width: 100%; height: 0; padding-bottom: 100%; border-radius: 8px; overflow: hidden; position: relative; background: #060612; }
#three-container canvas { position: absolute; top: 0; left: 0; width: 100% !important; height: 100% !important; display: block; }
.controls { display: flex; align-items: center; justify-content: center; gap: 8px; margin-top: 8px; flex-wrap: wrap; }
.controls button { background: #141428; color: #00ff88; border: 1px solid #00ff88; padding: 4px 12px; border-radius: 4px; cursor: pointer; font-size: 12px; transition: all 0.2s; }
.controls button:hover { background: #00ff88; color: #0a0a16; }
.controls input[type=range] { width: 120px; background: #1a1a30; height: 3px; -webkit-appearance: none; appearance: none; border-radius: 2px; }
.controls input[type=range]::-webkit-slider-thumb { -webkit-appearance: none; appearance: none; width: 12px; height: 12px; border-radius: 50%; background: #00ff88; cursor: pointer; }
.controls label { color: #888; font-size: 11px; display: flex; align-items: center; gap: 4px; }
.controls label input { accent-color: #00ff88; }
.status-bar { display: flex; justify-content: center; gap: 12px; margin-top: 8px; flex-wrap: wrap; }
.status-item { text-align: center; padding: 4px 10px; background: #0e0e1e; border-radius: 4px; border: 1px solid #1a1a30; min-width: 64px; }
.status-item .v { color: #00ff88; font-size: 15px; font-weight: 700; }
.status-item .v.amber { color: #ffaa00; }
.status-item .v.blue { color: #4488ff; }
.status-item .l { color: #666; font-size: 9px; text-transform: uppercase; letter-spacing: 0.5px; margin-top: 1px; }
.stats-row { display: flex; gap: 8px; margin-top: 8px; justify-content: center; }
.stats-row .stat-box { background: #0e0e1e; border-radius: 6px; padding: 8px 12px; border: 1px solid #1a1a30; text-align: center; flex: 1; }
.stats-row .stat-box .val { font-size: 18px; font-weight: 700; }
.stats-row .stat-box .lbl { font-size: 9px; color: #666; text-transform: uppercase; letter-spacing: 0.5px; }
#load-msg { color: #666; font-size: 13px; text-align: center; padding: 40px 20px; }
#error-msg { color: #ff4444; font-size: 13px; text-align: center; padding: 20px; display: none; }
@media (max-width: 900px) { .pipeline { flex-wrap: wrap; } .arrow-wrap { width: 100%; padding: 4px 0; } .arrow-wrap svg { transform: rotate(90deg); } }
</style>
</head>
<body>
<div class="header">
  <h1><span>RIPRA</span> Adaptive Optics Pipeline</h1>
  <p>REAL-TIME WAVEFRONT RECONSTRUCTION - TURBULENCE CHARACTERIZATION - DM MAPPING</p>
</div>
<div id="load-msg">Loading Three.js from CDN...</div>
<div id="error-msg">Failed to load 3D viewer. Check internet connection or browser console (F12).</div>
<div class="pipeline" id="pipeline-main" style="display:none">

<div class="stage">
  <div class="stage-title">SH-WFS Time-Series Input</div>
  <canvas id="spot-canvas" width="400" height="400"></canvas>
  <div class="status-bar">
    <div class="status-item"><div class="v">8x8</div><div class="l">Lenslets</div></div>
    <div class="status-item"><div class="v">648x492</div><div class="l">Camera (px)</div></div>
    <div class="status-item"><div class="v" id="spot-frame">0</div><div class="l">Frame</div></div>
  </div>
</div>

<div class="arrow-wrap">
  <svg viewBox="0 0 30 30">
    <defs><marker id="a" markerWidth="6" markerHeight="6" refX="26" refY="4" orient="auto">
      <path d="M0,0 L6,4 L0,8 Z" fill="#00ff88"/></marker></defs>
    <line x1="2" y1="15" x2="24" y2="15" stroke="#00ff88" stroke-width="2" marker-end="url(#a)"/>
    <circle cx="8" cy="15" r="2" fill="#00ff88" opacity="0.4"/>
    <circle cx="15" cy="15" r="2" fill="#00ff88" opacity="0.6"/>
  </svg>
</div>

<div class="stage">
  <div class="stage-title">Real-Time Processing Pipeline</div>
  <ul class="algo-list">
    <li>Iterative centroid estimation (TCoG)</li>
    <li>Zonal wavefront reconstruction</li>
    <li>Modal decomposition (Zernike)</li>
    <li>Turbulence: r0, tau0 estimation</li>
    <li>DM actuator command mapping</li>
  </ul>
  <div class="metrics">
    <div class="metric"><span class="label">Latency</span><span class="value">&lt; 1 ms/frame</span></div>
    <div class="metric"><span class="label">Throughput</span><span class="value">500 fps</span></div>
    <div class="metric"><span class="label">Stability</span><span class="value">sigma &lt; 0.05 lambda</span></div>
    <div class="metric"><span class="label">D/r0</span><span class="value amber" id="metric-dr0">--</span></div>
    <div class="metric"><span class="label">Regime</span><span class="value amber" id="metric-regime">--</span></div>
  </div>
</div>

<div class="arrow-wrap">
  <svg viewBox="0 0 30 30">
    <defs><marker id="b" markerWidth="6" markerHeight="6" refX="26" refY="4" orient="auto">
      <path d="M0,0 L6,4 L0,8 Z" fill="#00ff88"/></marker></defs>
    <line x1="2" y1="15" x2="24" y2="15" stroke="#00ff88" stroke-width="2" marker-end="url(#b)"/>
    <circle cx="8" cy="15" r="2" fill="#00ff88" opacity="0.4"/>
    <circle cx="15" cy="15" r="2" fill="#00ff88" opacity="0.6"/>
  </svg>
</div>

<div class="stage">
  <div class="stage-title">Reconstructed Wavefront</div>
  <div id="three-container"></div>
  <div class="controls">
    <button id="btn-play">Play</button>
    <button id="btn-prev">&#9664;</button>
    <input type="range" id="frame-slider" min="0" max="__NFRAMES__" value="0" step="1">
    <button id="btn-next">&#9654;</button>
    <label><input type="checkbox" id="chk-rotate" checked> Rotate</label>
  </div>
  <div class="stats-row">
    <div class="stat-box"><div class="val" id="stat-rms" style="color:#4488ff">0.000</div><div class="lbl">RMS (rad)</div></div>
    <div class="stat-box"><div class="val" id="stat-pv" style="color:#4488ff">0.000</div><div class="lbl">P-V (rad)</div></div>
    <div class="stat-box"><div class="val" id="stat-frame" style="color:#00ff88">0</div><div class="lbl">Frame</div></div>
  </div>
</div>

</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
<script>
(function() {
const COEFFS = __COEFFS__;
const D_R0 = __DR0__;
const N_FRAMES = __NFRAMES__;
const N_MODES = __NMODES__;

document.getElementById('load-msg').style.display = 'none';
document.getElementById('pipeline-main').style.display = 'flex';

var sc = document.getElementById('spot-canvas');
var sctx = sc.getContext('2d');

function drawSpots(frame) {
  var W = sc.width, H = sc.height, SP = 8;
  var seed = frame * 137.5;
  var rSpot = W / (SP * 2.6);
  var spacing = W / (SP + 1);
  var off = (W - spacing * (SP - 1)) / 2;
  sctx.fillStyle = '#060612'; sctx.fillRect(0, 0, W, H);
  for (var r = 0; r < SP; r++) {
    for (var c = 0; c < SP; c++) {
      var bx = off + c * spacing, by = off + r * spacing;
      var dx = Math.sin(seed + r * 0.7 + c * 1.3) * rSpot * 0.40;
      var dy = Math.cos(seed + r * 1.1 + c * 0.5) * rSpot * 0.40;
      var cx = bx + dx, cy = by + dy;
      var br = 0.55 + 0.35 * Math.sin(seed + r * 0.9 + c * 0.4);
      var bg = sctx.createRadialGradient(cx, cy, 0, cx, cy, rSpot * 0.9);
      bg.addColorStop(0, 'rgba('+Math.round(Math.min(255,255*br))+','+Math.round(Math.min(255,255*br))+','+Math.round(Math.min(255,200*br))+',1)');
      bg.addColorStop(0.15, 'rgba('+Math.round(Math.min(255,220*br))+','+Math.round(Math.min(255,255*br))+','+Math.round(Math.min(255,170*br))+',0.95)');
      bg.addColorStop(0.4, 'rgba('+Math.round(Math.min(255,140*br))+','+Math.round(Math.min(255,200*br))+','+Math.round(Math.min(255,100*br))+',0.6)');
      bg.addColorStop(0.7, 'rgba('+Math.round(Math.min(255,60*br))+','+Math.round(Math.min(255,100*br))+','+Math.round(Math.min(255,50*br))+',0.25)');
      bg.addColorStop(1, 'rgba(6,6,24,0)');
      sctx.fillStyle = bg; sctx.beginPath(); sctx.arc(cx, cy, rSpot * 0.9, 0, Math.PI * 2); sctx.fill();
      var sdx = Math.sin(seed * 0.5 + r * 0.4 + c * 1.1) * rSpot * 0.15;
      var sdy = Math.cos(seed * 0.4 + r * 1.2 + c * 0.7) * rSpot * 0.15;
      var sg = sctx.createRadialGradient(cx+sdx, cy+sdy, 0, cx+sdx, cy+sdy, rSpot*0.25);
      sg.addColorStop(0, 'rgba(255,255,255,0.7)');
      sg.addColorStop(1, 'rgba(255,255,255,0)');
      sctx.fillStyle = sg; sctx.beginPath(); sctx.arc(cx+sdx, cy+sdy, rSpot*0.25, 0, Math.PI*2); sctx.fill();
    }
  }
  document.getElementById('spot-frame').textContent = frame;
}

drawSpots(0);

var factCache = [1];
function fact(n) { while (factCache.length <= n) factCache.push(factCache.length * factCache[factCache.length - 1]); return factCache[n]; }
var ZF = [];
for (var k = 0; k < N_MODES; k++) {
  var j = k + 2, n = 0, m = 0, cj = 2;
  for (var rn = 1; rn < 20; rn++) {
    for (var rm = rn % 2; rm <= rn; rm += 2) {
      if (rm === 0) { if (cj === j) { n = rn; m = 0; } cj++; }
      else {
        if (cj % 2 === 1) { if (cj === j) { n = rn; m = -rm; } if (cj + 1 === j) { n = rn; m = rm; } }
        else { if (cj === j) { n = rn; m = rm; } if (cj + 1 === j) { n = rn; m = -rm; } }
        cj += 2;
      }
    }
  }
  var am = Math.abs(m), norm = m === 0 ? Math.sqrt(n + 1) : Math.sqrt(2 * (n + 1)), uc = m >= 0;
  var terms = [];
  for (var s = 0; s <= (n - am) / 2; s++) {
    var c = ((-1)**s * fact(n - s)) / (fact(s) * fact((n + am)/2 - s) * fact((n - am)/2 - s));
    terms.push({c: c, e: n - 2*s});
  }
  ZF.push({terms: terms, norm: norm, am: am, uc: uc});
}

function jetColor(t) {
  if (t < -0.75) return [0, 0, 0.5 + 2*(t+0.75)];
  if (t < -0.25) return [0, 2*(t+0.75), 1];
  if (t < 0.25) return [2*(t+0.25), 2*(t+0.25), 1 - 2*(t+0.25)];
  if (t < 0.75) return [1, 1 - 2*(t-0.25), 0];
  return [1 - 1.5*(t-0.75), 0.2 - 0.8*(t-0.75), 0];
}

function buildSurf(coeffs) {
  var N = 50, sc = 1.8;
  var pos = [], clr = [], idx = [];
  var Ncol = N;
  var vals = [];
  for (var i = 0; i < N; i++) vals[i] = [];
  var vmax = 0;
  for (var i = 0; i < N; i++) {
    for (var j = 0; j < N; j++) {
      var x = -1 + 2*j/(N-1), y = -1 + 2*i/(N-1);
      var rho = Math.sqrt(x*x + y*y);
      if (rho > 1) { vals[i][j] = NaN; continue; }
      var th = Math.atan2(y, x);
      var val = 0;
      for (var k = 0; k < N_MODES; k++) {
        var f = ZF[k]; var R = 0;
        for (var t = 0; t < f.terms.length; t++) R += f.terms[t].c * Math.pow(rho, f.terms[t].e);
        val += coeffs[k] * R * f.norm * (f.uc ? Math.cos(f.am * th) : Math.sin(f.am * th));
      }
      if (Math.abs(val) > vmax) vmax = Math.abs(val);
      vals[i][j] = val;
    }
  }
  vmax = vmax || 1;
  var im = [];
  for (var i = 0; i < N; i++) im[i] = [];
  var vi = 0;
  for (var i = 0; i < N; i++) {
    for (var j = 0; j < N; j++) {
      var v = vals[i][j];
      if (isNaN(v)) { im[i][j] = -1; continue; }
      im[i][j] = vi++;
      var px = (-1 + 2*j/(N-1))*sc, py = (-1 + 2*i/(N-1))*sc;
      var zScale = 0.5;
      pos.push(px, v * zScale, py);
      var t = v / vmax;
      var c = jetColor(t);
      clr.push(c[0], c[1], c[2]);
    }
  }
  for (var i = 0; i < N-1; i++) {
    for (var j = 0; j < N-1; j++) {
      var a = im[i][j], b = im[i][j+1], c = im[i+1][j], d = im[i+1][j+1];
      if (a < 0 || b < 0 || c < 0 || d < 0) continue;
      idx.push(a, b, c, b, d, c);
    }
  }
  return {pos: pos, clr: clr, idx: idx, vmax: vmax, vals: vals, N: N};
}

var container = document.getElementById('three-container');
var W = container.offsetWidth || 350;
if (W < 50) W = 350;

var scene = new THREE.Scene();
scene.background = new THREE.Color(0x060612);

var camera = new THREE.PerspectiveCamera(35, 1, 0.1, 100);
camera.position.set(3.8, 2.8, 3.8);

var renderer = new THREE.WebGLRenderer({antialias: true});
renderer.setSize(W, W);
renderer.setPixelRatio(window.devicePixelRatio || 1);
container.appendChild(renderer.domElement);

var controls = new THREE.OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.target.set(0, 0.15, 0);
controls.autoRotate = true;
controls.autoRotateSpeed = 1.2;
controls.update();

var amb = new THREE.AmbientLight(0x404060);
scene.add(amb);
var dl = new THREE.DirectionalLight(0xffffff, 0.85);
dl.position.set(5, 8, 5);
scene.add(dl);
var fl = new THREE.DirectionalLight(0x4488ff, 0.3);
fl.position.set(-4, 2, 4);
scene.add(fl);
var gh = new THREE.GridHelper(5, 10, 0x00cc66, 0x1a1a3a);
gh.position.y = -0.85;
scene.add(gh);

var mesh = null, cf = 0, playing = false, timer = null;

function makeTextSprite(msg, color) {
  var canvas = document.createElement('canvas');
  canvas.width = 128; canvas.height = 48;
  var ctx = canvas.getContext('2d');
  ctx.fillStyle = 'rgba(0,0,0,0)'; ctx.fillRect(0, 0, 128, 48);
  ctx.font = 'Bold 20px Arial';
  ctx.textAlign = 'center';
  ctx.fillStyle = color || '#00ff88';
  ctx.fillText(msg, 64, 30);
  var tex = new THREE.CanvasTexture(canvas);
  var mat = new THREE.SpriteMaterial({map: tex, transparent: true, depthTest: false, depthWrite: false});
  var sprite = new THREE.Sprite(mat);
  sprite.scale.set(1.0, 0.35, 1);
  return sprite;
}

var labels = [];
function addLabel(text, pos, color) {
  var s = makeTextSprite(text, color);
  s.position.set(pos[0], pos[1], pos[2]);
  scene.add(s);
  labels.push(s);
}
addLabel('X (lambda)', [2.3, -0.85, 0], '#00ff88');
addLabel('Y (lambda)', [0, -0.85, 2.3], '#00ff88');
addLabel('Phase', [0, 1.8, 0], '#4488ff');
addLabel('-pi', [-2.1, -0.85, 0], '#888');
addLabel('+pi', [2.2, -0.85, 0], '#888');

function updateFrame() {
  if (mesh) { scene.remove(mesh); }
  drawSpots(cf);
  var data = buildSurf(COEFFS[cf]);
  
  var geom = new THREE.BufferGeometry();
  geom.setAttribute('position', new THREE.Float32BufferAttribute(data.pos, 3));
  geom.setAttribute('color', new THREE.Float32BufferAttribute(data.clr, 3));
  geom.setIndex(data.idx);
  geom.computeVertexNormals();
  
  var mat = new THREE.MeshPhongMaterial({
    vertexColors: true, side: THREE.DoubleSide,
    specular: 0x222244, shininess: 25, transparent: true, opacity: 0.94
  });
  mesh = new THREE.Mesh(geom, mat);
  scene.add(mesh);

  var dr0 = D_R0[cf];
  var regime = dr0 < 3 ? 'Weak' : dr0 < 6 ? 'Moderate' : 'Strong';
  document.getElementById('metric-dr0').textContent = dr0.toFixed(2);
  document.getElementById('metric-regime').textContent = regime;
  document.getElementById('stat-frame').textContent = cf;
  var sq = 0, cnt = 0, mn = Infinity, mx = -Infinity;
  for (var i = 0; i < data.vals.length; i++) {
    for (var j = 0; j < data.vals[i].length; j++) {
      var v = data.vals[i][j];
      if (!isNaN(v)) { sq += v*v; cnt++; if (v < mn) mn = v; if (v > mx) mx = v; }
    }
  }
  document.getElementById('stat-rms').textContent = Math.sqrt(sq/cnt).toFixed(4);
  document.getElementById('stat-pv').textContent = (mx - mn).toFixed(4);
  document.getElementById('frame-slider').value = cf;
}

function animate() {
  controls.update();
  renderer.render(scene, camera);
  requestAnimationFrame(animate);
}

updateFrame();
animate();

window.addEventListener('resize', function() {
  var w = container.offsetWidth;
  if (w < 50) w = 350;
  renderer.setSize(w, w);
});

document.getElementById('chk-rotate').onchange = function() { controls.autoRotate = this.checked; };
document.getElementById('btn-play').onclick = function() {
  if (playing) { playing = false; document.getElementById('btn-play').textContent = 'Play'; clearInterval(timer); }
  else { playing = true; document.getElementById('btn-play').textContent = 'Stop';
    timer = setInterval(function() { cf = (cf+1) % N_FRAMES; updateFrame(); }, 80); }
};
document.getElementById('btn-prev').onclick = function() { cf = (cf-1+N_FRAMES)%N_FRAMES; if(playing){playing=0;clearInterval(timer);document.getElementById('btn-play').textContent='Play';} updateFrame(); };
document.getElementById('btn-next').onclick = function() { cf = (cf+1)%N_FRAMES; if(playing){playing=0;clearInterval(timer);document.getElementById('btn-play').textContent='Play';} updateFrame(); };
document.getElementById('frame-slider').oninput = function() { cf = parseInt(this.value); if(playing){playing=0;clearInterval(timer);document.getElementById('btn-play').textContent='Play';} updateFrame(); };

})();
</script>
</body>
</html>"""

    html = html.replace('__COEFFS__', coeffs_json)
    html = html.replace('__DR0__', dr0_json)
    html = html.replace('__NFRAMES__', str(n_frames))
    html = html.replace('__NMODES__', str(nmodes))

    html_path = os.path.join(OUT, 'pipeline_dashboard.html')
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html)
    size_kb = os.path.getsize(html_path) / 1024
    print(f"Saved pipeline_dashboard.html ({size_kb:.0f} KB)")
    print(f"  {n_frames} frames, {nmodes} Zernike modes embedded")

if __name__ == '__main__':
    generate()
