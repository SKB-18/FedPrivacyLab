"""
TensorFlow Lite model quantization utilities.

Supports int8 and float16 post-training quantization with benchmark tracking.
Privacy note: quantization is applied to the aggregated global model only;
no individual client data is exposed during conversion.
"""

from __future__ import annotations

import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    import tensorflow as tf
    _HAS_TF = True
except ImportError:
    _HAS_TF = False


QUANTIZATION_TYPES = ("int8", "float16", "dynamic")


@dataclass
class QuantizationBenchmark:
    """Tracks size, accuracy, and latency impact of quantization."""
    quantization_type: str
    original_size_mb: float
    quantized_size_mb: float
    size_reduction_percent: float
    original_accuracy: float
    quantized_accuracy: float
    accuracy_drop_percent: float
    inference_latency_ms: float
    original_latency_ms: float


def get_model_size(model_path: str) -> float:
    """
    Return the on-disk size of a model file in megabytes.

    Args:
        model_path: Path to a TFLite flatbuffer (.tflite) or SavedModel directory.

    Returns:
        Size in MB.
    """
    p = Path(model_path)
    if p.is_file():
        return p.stat().st_size / (1024 ** 2)
    # SavedModel directory
    total = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
    return total / (1024 ** 2)


def _save_model_temp(model: "tf.keras.Model") -> str:
    """Save a Keras model to a temp directory and return the path."""
    import tensorflow as tf
    tmp = tempfile.mkdtemp(prefix="fedprivacylab_")
    save_path = os.path.join(tmp, "saved_model")
    # Keras 3+ requires explicit SavedModel export for TFLite conversion
    try:
        model.export(save_path)
    except AttributeError:
        # Older Keras: model.save with a directory path works fine
        model.save(save_path)
    return save_path


def _representative_dataset_gen(X_sample: np.ndarray):
    """Generator for TFLite int8 calibration dataset."""
    for row in X_sample[:100]:
        yield [row[np.newaxis, :].astype(np.float32)]


def quantize_model(
    model: "tf.keras.Model",
    quantization_type: str = "int8",
    calibration_data: np.ndarray | None = None,
    output_path: str | None = None,
) -> bytes:
    """
    Convert a Keras model to TFLite with the specified quantization scheme.

    Args:
        model: Trained tf.keras.Model (the aggregated global model).
        quantization_type: One of 'int8', 'float16', or 'dynamic'.
            - 'int8': Full integer quantization (requires calibration_data).
            - 'float16': Half-precision weights, float32 activations.
            - 'dynamic': Dynamic range quantization (no calibration needed).
        calibration_data: Representative numpy array (n_samples, n_features)
            used for int8 calibration. Required when quantization_type='int8'.
        output_path: If provided, save the .tflite flatbuffer here.

    Returns:
        TFLite flatbuffer as bytes.

    Raises:
        RuntimeError: If TensorFlow is not installed.
        ValueError: If quantization_type is unknown.
    """
    if not _HAS_TF:
        raise RuntimeError("TensorFlow is required for model quantization.")
    if quantization_type not in QUANTIZATION_TYPES:
        raise ValueError(f"quantization_type must be one of {QUANTIZATION_TYPES}")

    save_path = _save_model_temp(model)
    converter = tf.lite.TFLiteConverter.from_saved_model(save_path)

    if quantization_type == "int8":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        if calibration_data is not None:
            converter.representative_dataset = lambda: _representative_dataset_gen(calibration_data)

    elif quantization_type == "float16":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]

    elif quantization_type == "dynamic":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

    tflite_model = converter.convert()

    if output_path:
        with open(output_path, "wb") as f:
            f.write(tflite_model)

    return tflite_model


def _run_tflite_inference(tflite_model: bytes, X: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Run TFLite inference on X.

    Returns:
        (predictions array, latency_ms per sample)
    """
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    in_detail = interpreter.get_input_details()[0]
    out_detail = interpreter.get_output_details()[0]

    preds = []
    t0 = time.perf_counter()
    for row in X:
        inp = row[np.newaxis, :].astype(in_detail["dtype"])
        interpreter.set_tensor(in_detail["index"], inp)
        interpreter.invoke()
        out = interpreter.get_tensor(out_detail["index"])
        preds.append(float(out.ravel()[0]))
    elapsed_ms = (time.perf_counter() - t0) * 1000
    return np.array(preds), elapsed_ms / max(len(X), 1)


def evaluate_quantized_model(
    original_model: "tf.keras.Model",
    quantized_model: bytes,
    test_data: tuple[np.ndarray, np.ndarray],
    quantization_type: str = "int8",
) -> QuantizationBenchmark:
    """
    Compare a full Keras model against its quantized TFLite counterpart.

    Args:
        original_model: The unquantized Keras model.
        quantized_model: TFLite flatbuffer bytes (output of quantize_model()).
        test_data: Tuple of (X_test, y_test) numpy arrays.
        quantization_type: Label for the benchmark record.

    Returns:
        QuantizationBenchmark dataclass with size/accuracy/latency metrics.
    """
    if not _HAS_TF:
        raise RuntimeError("TensorFlow is required.")

    X_test, y_test = test_data
    X_test = X_test.astype(np.float32)

    # Original model metrics
    t0 = time.perf_counter()
    orig_preds_raw = original_model.predict(X_test, verbose=0).ravel()
    orig_latency_ms = (time.perf_counter() - t0) * 1000 / max(len(X_test), 1)
    orig_preds = (orig_preds_raw > 0.5).astype(int)
    orig_acc = float(np.mean(orig_preds == y_test))

    # Quantized model metrics
    quant_preds_raw, quant_latency_ms = _run_tflite_inference(quantized_model, X_test)
    # int8 models output int8; scale to [0,1] by checking sign
    if quantization_type == "int8":
        quant_preds = (quant_preds_raw > 0).astype(int)
    else:
        quant_preds = (quant_preds_raw > 0.5).astype(int)
    quant_acc = float(np.mean(quant_preds == y_test))

    # Size comparison
    orig_size_mb = get_model_size(_save_model_temp(original_model))
    with tempfile.NamedTemporaryFile(suffix=".tflite", delete=False) as f:
        f.write(quantized_model)
        tmp_path = f.name
    quant_size_mb = get_model_size(tmp_path)
    os.unlink(tmp_path)

    acc_drop = (orig_acc - quant_acc) / max(orig_acc, 1e-9) * 100

    return QuantizationBenchmark(
        quantization_type=quantization_type,
        original_size_mb=round(orig_size_mb, 3),
        quantized_size_mb=round(quant_size_mb, 3),
        size_reduction_percent=round((1 - quant_size_mb / orig_size_mb) * 100, 1),
        original_accuracy=round(orig_acc, 4),
        quantized_accuracy=round(quant_acc, 4),
        accuracy_drop_percent=round(acc_drop, 2),
        inference_latency_ms=round(quant_latency_ms, 3),
        original_latency_ms=round(orig_latency_ms, 3),
    )
