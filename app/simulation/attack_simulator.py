"""Simulate malicious or noisy client behavior for risk analysis."""

import numpy as np


def inject_poisoned_update(
    update: np.ndarray, scale: float = 10.0, seed: int = 0
) -> np.ndarray:
    """Scale update dramatically to simulate model poisoning."""
    rng = np.random.default_rng(seed)
    direction = rng.normal(0, 1, size=update.shape)
    direction = direction / (np.linalg.norm(direction) + 1e-8)
    return update + scale * direction


def describe_attack_risks() -> list[dict[str, str]]:
    return [
        {
            "risk": "Model poisoning",
            "description": "Malicious clients send scaled updates",
            "mitigation": "Update clipping, secure aggregation, anomaly detection",
        },
        {
            "risk": "Update leakage",
            "description": "Individual gradients may reveal training data",
            "mitigation": "Secure aggregation simulation, DP noise, clipping",
        },
        {
            "risk": "Aggregate inference",
            "description": "Rare cohorts identifiable in analytics",
            "mitigation": "Cohort suppression (min 100 clients), DP on counts",
        },
    ]
