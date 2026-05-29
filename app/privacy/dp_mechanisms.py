"""Differential privacy noise mechanisms."""

import numpy as np


def laplace_count(count: float, sensitivity: float, epsilon: float) -> float:
    """Add Laplace noise to a bounded count metric."""
    if epsilon <= 0:
        return count
    scale = sensitivity / epsilon
    return count + float(np.random.laplace(0, scale))


def gaussian_sum_noise(
    value: float, sensitivity: float, epsilon: float, delta: float = 1e-6
) -> float:
    """Add Gaussian noise to a bounded sum (simplified DP for sums)."""
    if epsilon <= 0:
        return value
    sigma = sensitivity * np.sqrt(2 * np.log(1.25 / delta)) / epsilon
    return value + float(np.random.normal(0, sigma))


def add_gaussian_noise_to_update(
    update: np.ndarray,
    clipping_norm: float,
    noise_multiplier: float,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Add Gaussian DP noise to aggregate model update after clipping."""
    gen = rng if rng is not None else np.random.default_rng()
    noise = gen.normal(
        loc=0.0,
        scale=noise_multiplier * clipping_norm,
        size=update.shape,
    )
    return update + noise


def noisy_average_from_sum(
    noisy_sum: float, count: int, epsilon: float, delta: float = 1e-6
) -> float:
    """Compute noisy average from noisy sum and bounded count."""
    if count <= 0:
        return 0.0
    return noisy_sum / count
