"""
rippra/bindings/onnx_inference.py - ONNX Runtime inference wrapper

Loads exported ONNX models and runs inference. Falls back to PyTorch if
ONNX Runtime is unavailable.
"""

import os
import numpy as np

try:
    import onnxruntime as ort
    HAVE_ORT = True
except ImportError:
    HAVE_ORT = False


class ONNXInference:
    """Run inference on exported ONNX models."""

    def __init__(self, model_path, providers=None):
        if not HAVE_ORT:
            raise ImportError("onnxruntime not installed. Install with: pip install onnxruntime")
        if providers is None:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if ort.get_device() == 'GPU' else ['CPUExecutionProvider']
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.output_shape = self.session.get_outputs()[0].shape

    def run(self, input_array):
        if input_array.ndim == 1:
            input_array = input_array[np.newaxis, :]
        elif input_array.ndim == 3:
            input_array = input_array[np.newaxis, :, :, :]
        input_array = input_array.astype(np.float32)
        outputs = self.session.run([self.output_name], {self.input_name: input_array})
        return outputs[0].squeeze()


def load_model(model_name='wavefront_cnn'):
    """Convenience: load an ONNX model from the onnx_models directory.

    Args:
        model_name: 'wavefront_mlp', 'wavefront_cnn', or 'wavefront_lstm'

    Returns:
        ONNXInference instance or None if model file not found.
    """
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    onnx_dir = os.path.join(base, "onnx_models")
    path = os.path.join(onnx_dir, f"{model_name}.onnx")
    if not os.path.exists(path):
        return None
    return ONNXInference(path)
