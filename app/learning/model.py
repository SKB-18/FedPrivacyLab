"""Keras binary classifier for poor experience prediction."""

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

FEATURE_COLUMNS = [
    "latency_ms",
    "failure_count",
    "click_count",
    "session_length_sec",
    "device_tier_enc",
    "network_type_enc",
    "app_version_enc",
    "locale_bucket_enc",
]

CATEGORICAL_MAPS: dict[str, dict[str, int]] = {
    "device_tier": {"low": 0, "mid": 1, "high": 2},
    "network_type": {"wifi": 0, "cellular": 1, "offline": 2},
    "app_version": {"1.1": 0, "1.2": 1, "1.3": 2, "1.4": 3},
    "locale_bucket": {
        "en_US": 0,
        "en_GB": 1,
        "de_DE": 2,
        "fr_FR": 3,
        "ja_JP": 4,
        "es_ES": 5,
    },
}

INPUT_DIM = len(FEATURE_COLUMNS)


def encode_record_aggregate(records: list[dict]) -> np.ndarray:
    """Aggregate client records into a single feature vector for training."""
    if not records:
        return np.zeros(INPUT_DIM, dtype=np.float32)

    latency = np.mean([float(r.get("latency_ms", 0)) for r in records])
    failures = sum(int(r.get("failure", 0)) for r in records)
    clicks = sum(int(r.get("clicked_recommendation", 0)) for r in records)
    session = np.mean([float(r.get("session_length_sec", 0)) for r in records])

    device = records[0].get("device_tier", "mid")
    network = records[0].get("network_type", "wifi")
    version = records[0].get("app_version", "1.3")
    locale = records[0].get("locale_bucket", records[0].get("locale", "en_US"))

    return np.array(
        [
            latency,
            failures,
            clicks,
            session,
            CATEGORICAL_MAPS["device_tier"].get(device, 1),
            CATEGORICAL_MAPS["network_type"].get(network, 0),
            CATEGORICAL_MAPS["app_version"].get(version, 2),
            CATEGORICAL_MAPS["locale_bucket"].get(locale, 0),
        ],
        dtype=np.float32,
    )


def records_to_xy(records: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Convert client records to feature matrix and labels (delegates to feature_encoding)."""
    from app.learning.feature_encoding import records_to_xy as _records_to_xy

    return _records_to_xy(records, normalize=True)


def initial_global_weights(seed: int, input_dim: int = INPUT_DIM) -> np.ndarray:
    """Reproducible starting weights for federated learning (same seed → same vector)."""
    tf.keras.backend.clear_session()
    tf.random.set_seed(seed)
    model = build_model(input_dim=input_dim)
    return weights_to_vector(model)


def build_model(input_dim: int = INPUT_DIM) -> keras.Model:
    """Dense(32, relu) -> Dense(16, relu) -> Dense(1, sigmoid)."""
    model = keras.Sequential(
        [
            layers.Input(shape=(input_dim,)),
            layers.Dense(32, activation="relu"),
            layers.Dense(16, activation="relu"),
            layers.Dense(1, activation="sigmoid"),
        ],
        name="poor_experience_classifier",
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def weights_to_vector(model: keras.Model) -> np.ndarray:
    return np.concatenate([w.flatten() for w in model.get_weights()])


def vector_to_weights(vector: np.ndarray, model: keras.Model) -> list[np.ndarray]:
    shapes = [w.shape for w in model.get_weights()]
    splits = []
    offset = 0
    for shape in shapes:
        size = int(np.prod(shape))
        splits.append(vector[offset : offset + size].reshape(shape))
        offset += size
    return splits


def set_model_weights(model: keras.Model, vector: np.ndarray) -> None:
    model.set_weights(vector_to_weights(vector, model))


def compute_weight_delta(
    global_vector: np.ndarray, local_vector: np.ndarray
) -> np.ndarray:
    return local_vector - global_vector
