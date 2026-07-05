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

# export_onnx.py only exports MLP and CNN (LSTM exported separately)
model_files = ["wavefront_mlp.onnx", "wavefront_cnn.onnx"]
all_ok = True

for fname in model_files:
    path = os.path.join(ONNX_DIR, fname)
    if not os.path.exists(path):
        print(f"FAIL: {fname} not found")
        all_ok = False
        continue

    try:
        sess = ort.InferenceSession(path)
        inp = sess.get_inputs()[0]
        out = sess.get_outputs()[0]
        inp_shape = list(inp.shape)
        out_shape = list(out.shape)

        # Verify output has dynamic batch dim and 20 modes
        out_ok = len(out_shape) == 2 and out_shape[1] == 20

        # Run inference with random data matching input shape
        dummy = np.random.randn(*[s if isinstance(s, int) else 1 for s in inp_shape]).astype(np.float32)
        result = sess.run(None, {inp.name: dummy})
        actual_out = list(result[0].shape)

        run_ok = actual_out[1] == 20
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
