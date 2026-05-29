"""Unit tests for secure aggregation simulation."""

import numpy as np

from app.privacy.secure_aggregation import (
    aggregate_masked_updates,
    mask_update,
    secure_aggregate_updates,
)


def test_masks_cancel():
    u1 = np.array([1.0, 2.0, 3.0])
    u2 = np.array([0.5, -1.0, 2.0])
    m1, mask1 = mask_update(u1, seed=1)
    m2, mask2 = mask_update(u2, seed=2)
    agg = aggregate_masked_updates([m1, m2], [mask1, mask2])
    np.testing.assert_allclose(agg, u1 + u2, rtol=1e-5)


def test_secure_aggregate_updates():
    updates = [np.array([1.0, 0.0]), np.array([2.0, 1.0])]
    result = secure_aggregate_updates(updates, base_seed=42)
    np.testing.assert_allclose(result, np.array([3.0, 1.0]), rtol=1e-5)
