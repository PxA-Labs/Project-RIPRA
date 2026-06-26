"""Verify all ONNX models load and produce correct output shapes."""
import os, sys

try:
    import onnxruntime as ort
except ImportError:
    print("SKIP: onnxruntime not installed")
    sys.exit(0)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ONNX_DIR = os.path.join(BASE, "onnx_models")

models = {
    "wavefront_mlp.onnx":  ([1, 254],        [1, 20]),
    "wavefront_cnn.onnx":  ([1, 2, 11, 13],  [1, 20]),
    "wavefront_lstm.onnx": ([1, 10, 20],     [1, 20]),
}

all_ok = True
for fname, (in_shape, out_shape) in models.items():
    path = os.path.join(ONNX_DIR, fname)
    if not os.path.exists(path):
        print(f"FAIL: {fname} not found")
        all_ok = False
        continue
    try:
        sess = ort.InferenceSession(path)
        actual_in = list(sess.get_inputs()[0].shape)
        actual_out = list(sess.get_outputs()[0].shape)
        # Check dynamic batch dim (allow batch_size string)
        in_match = all(
            a == b or isinstance(a, str) or isinstance(b, str)
            for a, b in zip(actual_in, in_shape)
        )
        out_match = all(
            a == b or isinstance(a, str) or isinstance(b, str)
            for a, b in zip(actual_out, out_shape)
        )
        status = "OK" if (in_match and out_match) else "SHAPE MISMATCH"
        print(f"  {status}: {fname}  {actual_in} -> {actual_out}")
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
