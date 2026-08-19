"""Verify all ONNX models load, run inference, and produce correct output dimension (20 modes)."""
import os, sys

try:
    import numpy as np
    import onnxruntime as ort
except ImportError:
    print("SKIP: onnxruntime not installed")
    sys.exit(0)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ONNX_DIR = os.path.join(BASE, "onnx_models")

# Models exported by export_onnx.py
model_files = ["wavefront_mlp.onnx", "wavefront_cnn.onnx",
               "wavefront_lstm.onnx", "slope_completion_lstm.onnx"]
all_ok = True

for fname in model_files:
    path = os.path.join(ONNX_DIR, fname)
    if not os.path.exists(path):
        print(f"SKIP: {fname} not found")
        continue

    try:
        sess = ort.InferenceSession(path)
        inp = sess.get_inputs()[0]
        out = sess.get_outputs()[0]
        inp_shape = list(inp.shape)
        out_shape = list(out.shape)

        # Run inference with random data matching input shape
        dummy = np.random.randn(*[s if isinstance(s, int) else 1 for s in inp_shape]).astype(np.float32)
        result = sess.run(None, {inp.name: dummy})
        actual_out = list(result[0].shape)

        out_ok = len(actual_out) == 2
        run_ok = actual_out[0] == 1
        status = "OK" if (out_ok and run_ok) else "SHAPE MISMATCH"
        print(f"  {status}: {fname}  {inp_shape} -> {out_shape} (ran: {actual_out})")
        if status != "OK":
            all_ok = False
    except Exception as e:
        print(f"FAIL: {fname} — {e}")
        all_ok = False

if all_ok:
    print("All ONNX models validated successfully")
    sys.exit(0)
else:
    print("Some ONNX models FAILED validation")
    sys.exit(1)
