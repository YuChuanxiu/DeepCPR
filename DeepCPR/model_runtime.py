"""Runtime-neutral model loading for TensorFlow and ONNX models."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def tensorflow_available() -> bool:
    """Return whether TensorFlow can be imported in the current environment."""
    try:
        import tensorflow  # noqa: F401
    except Exception:
        return False
    return True


class ONNXModel:
    """Small Keras-like adapter around an ONNX Runtime inference session."""

    def __init__(self, modelpath: str | Path):
        try:
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "ONNX inference requires the 'onnxruntime' package. "
                "Install requirements-onnx.txt or install onnxruntime separately."
            ) from exc

        self.modelpath = str(modelpath)
        self.session = ort.InferenceSession(
            self.modelpath,
            providers=["CPUExecutionProvider"],
        )
        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        if len(inputs) != 1 or not outputs:
            raise ValueError(
                f"Expected one ONNX input and at least one output, got "
                f"{len(inputs)} inputs and {len(outputs)} outputs."
            )
        self.input_name = inputs[0].name
        self.output_names = [output.name for output in outputs]
        self.input_shape = inputs[0].shape
        self.output_shape = outputs[0].shape

    def predict(self, batch, verbose=0, batch_size=None):
        """Match the subset of the Keras ``predict`` API used by DeepCPR."""
        del verbose
        value = np.asarray(batch, dtype=np.float32)
        if batch_size is None or int(batch_size) <= 0 or value.shape[0] <= int(batch_size):
            return self.session.run(
                self.output_names, {self.input_name: value}
            )[0]
        outputs = []
        for start in range(0, value.shape[0], int(batch_size)):
            outputs.append(self.session.run(
                self.output_names,
                {self.input_name: value[start:start + int(batch_size)]},
            )[0])
        return np.concatenate(outputs, axis=0)


def _sibling_onnx_path(modelpath: str | Path) -> Path:
    path = Path(modelpath)
    return path.with_suffix(".onnx")


def load_model_auto(modelpath: str | Path, custom_objects=None):
    """Load a TensorFlow or ONNX model using the available runtime.

    An explicit ``.onnx`` path always selects ONNX Runtime.  For an H5/Keras
    path, TensorFlow is preferred when importable; if it is unavailable, a
    sibling file with the same stem and an ``.onnx`` suffix is used.
    """
    path = Path(modelpath)
    suffix = path.suffix.lower()
    if suffix == ".onnx":
        return ONNXModel(path)

    if tensorflow_available():
        import tensorflow as tf

        return tf.keras.models.load_model(
            str(path), custom_objects=custom_objects or {}
        )

    fallback = _sibling_onnx_path(path)
    if fallback.exists():
        return ONNXModel(fallback)

    raise ImportError(
        f"TensorFlow is unavailable and no ONNX fallback was found for '{path}'. "
        f"Pass the .onnx model path explicitly or place '{fallback.name}' beside it."
    )
