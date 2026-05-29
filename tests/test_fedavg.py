"""Unit tests for FedAvg."""

import numpy as np

from app.learning.fedavg import aggregate_client_deltas, fedavg


def test_fedavg_weighted_average():
    global_w = np.array([1.0, 2.0, 3.0])
    u1 = np.array([0.1, 0.0, -0.1])
    u2 = np.array([0.2, 0.1, 0.0])
    result = fedavg(global_w, [u1, u2], [100, 300])
    expected = global_w + (100 / 400) * u1 + (300 / 400) * u2
    np.testing.assert_allclose(result, expected)


def test_fedavg_empty_updates():
    global_w = np.array([1.0, 2.0])
    result = fedavg(global_w, [], [])
    np.testing.assert_array_equal(result, global_w)


def test_aggregate_deltas():
    d1 = np.array([1.0, 0.0])
    d2 = np.array([3.0, 0.0])
    agg = aggregate_client_deltas([d1, d2], [1, 3])
    np.testing.assert_allclose(agg, np.array([2.5, 0.0]))
