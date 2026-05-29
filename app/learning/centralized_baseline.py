"""Centralized training baseline for utility comparison."""

import numpy as np
import tensorflow as tf

from app.learning.evaluator import evaluate_model
from app.learning.model import build_model, records_to_xy, weights_to_vector


def train_centralized_baseline(
    all_records: list[dict],
    epochs: int = 10,
    validation_split: float = 0.2,
    seed: int = 42,
) -> tuple[tf.keras.Model, dict]:
    """Train model on all data centrally (upper bound on utility)."""
    tf.random.set_seed(seed)
    X, y = records_to_xy(all_records)
    if len(X) == 0:
        model = build_model()
        return model, {"accuracy": 0.0, "roc_auc": 0.0}

    model = build_model()
    class_weight = None
    if len(y) > 0:
        pos = float(y.sum())
        neg = len(y) - pos
        if pos > 0 and neg > 0:
            class_weight = {0: 1.0, 1: min(neg / pos, 10.0)}
    model.fit(
        X,
        y,
        epochs=epochs,
        batch_size=32,
        validation_split=validation_split,
        verbose=0,
        class_weight=class_weight,
    )
    metrics = evaluate_model(model, X, y)
    metrics["final_weights_norm"] = float(np.linalg.norm(weights_to_vector(model)))
    return model, metrics
