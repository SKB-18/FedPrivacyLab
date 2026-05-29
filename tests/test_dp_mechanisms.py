"""Unit tests for DP mechanisms."""

import numpy as np

from app.privacy.dp_mechanisms import (
    add_gaussian_noise_to_update,
    laplace_count,
)


def test_laplace_count_changes_value():
    np.random.seed(0)
    noisy = laplace_count(100.0, sensitivity=1.0, epsilon=1.0)
    assert noisy != 100.0


def test_laplace_zero_epsilon():
    assert laplace_count(50.0, 1.0, 0.0) == 50.0


def test_gaussian_noise_shape():
    update = np.zeros(10)
    noisy = add_gaussian_noise_to_update(update, clipping_norm=1.0, noise_multiplier=0.5)
    assert noisy.shape == (10,)
