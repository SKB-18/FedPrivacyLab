"""Feature encoding and normalization for stable training on real + synthetic data."""

import numpy as np

from app.learning.model import CATEGORICAL_MAPS, INPUT_DIM

# Global normalization stats (fit once per experiment from local client samples)
_norm_stats: dict[str, np.ndarray] | None = None


def record_to_feature_vector(record: dict) -> np.ndarray:
    """Per-event feature vector matching spec input fields."""
    device = record.get("device_tier", "mid")
    network = record.get("network_type", "wifi")
    version = record.get("app_version", "1.3")
    locale = record.get("locale_bucket", record.get("locale", "en_US"))

    return np.array(
        [
            float(record.get("latency_ms", 0)),
            float(record.get("failure", 0)),
            float(record.get("clicked_recommendation", 0)),
            float(record.get("session_length_sec", 0)),
            float(CATEGORICAL_MAPS["device_tier"].get(device, 1)),
            float(CATEGORICAL_MAPS["network_type"].get(network, 0)),
            float(CATEGORICAL_MAPS["app_version"].get(version, 2)),
            float(CATEGORICAL_MAPS["locale_bucket"].get(locale, 0)),
        ],
        dtype=np.float32,
    )


def fit_normalization(records: list[dict]) -> None:
    """Fit z-score normalization on continuous columns (latency, session)."""
    global _norm_stats
    if not records:
        _norm_stats = None
        return
    X = np.stack([record_to_feature_vector(r) for r in records])
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std < 1e-6] = 1.0
    _norm_stats = {"mean": mean, "std": std}


def normalize_features(X: np.ndarray) -> np.ndarray:
    if _norm_stats is None or len(X) == 0:
        return X
    return (X - _norm_stats["mean"]) / _norm_stats["std"]


def records_to_xy(records: list[dict], normalize: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Convert records to (X, y) with optional normalization."""
    if not records:
        return np.zeros((0, INPUT_DIM)), np.zeros((0, 1))
    X = np.stack([record_to_feature_vector(r) for r in records])
    if normalize and _norm_stats is not None:
        X = normalize_features(X)
    y = np.array(
        [[int(r.get("label_poor_experience", 0))] for r in records], dtype=np.float32
    )
    return X, y


def reset_normalization() -> None:
    global _norm_stats
    _norm_stats = None
