"""Client-side local model training."""

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf

from app.learning.model import (
    build_model,
    compute_weight_delta,
    records_to_xy,
    set_model_weights,
    weights_to_vector,
)


def train_local_model(
    records: list[dict],
    global_weights: np.ndarray,
    local_epochs: int = 2,
    seed: int = 0,
) -> tuple[np.ndarray, float, int]:
    """
    Train on local data starting from global weights.
    Returns (weight_delta, update_norm, num_examples).
    """
    tf.random.set_seed(seed)
    X, y = records_to_xy(records)
    n = len(X)
    if n == 0:
        return np.zeros_like(global_weights), 0.0, 0

    model = build_model()
    set_model_weights(model, global_weights)
    class_weight = None
    if len(y) > 0:
        pos = float(y.sum())
        neg = len(y) - pos
        if pos > 0 and neg > 0:
            class_weight = {0: 1.0, 1: min(neg / pos, 10.0)}
    model.fit(
        X,
        y,
        epochs=local_epochs,
        batch_size=min(32, n),
        verbose=0,
        class_weight=class_weight,
    )

    local_vector = weights_to_vector(model)
    delta = compute_weight_delta(global_weights, local_vector)
    norm = float(np.linalg.norm(delta))
    return delta, norm, n
