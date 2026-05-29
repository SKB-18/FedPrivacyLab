"""Unit tests for update clipping."""

import numpy as np

from app.privacy.update_clipping import clip_update


def test_no_clip_when_under_norm():
    v = np.array([0.1, 0.2, 0.3])
    clipped, was = clip_update(v, max_norm=10.0)
    np.testing.assert_allclose(clipped, v)
    assert was is False


def test_clip_when_over_norm():
    v = np.array([3.0, 4.0])
    clipped, was = clip_update(v, max_norm=1.0)
    assert was is True
    assert np.linalg.norm(clipped) <= 1.0 + 1e-6
