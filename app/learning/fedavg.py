"""FedAvg implementation."""

import numpy as np


def fedavg(
    global_weights: np.ndarray,
    client_updates: list[np.ndarray],
    client_sizes: list[int],
) -> np.ndarray:
    """
    Weighted average of client updates applied to global weights.
    global_update = sum((n_i / N) * client_update_i)
    new_global = global_weights + global_update
    """
    if not client_updates:
        return global_weights.copy()
    total_examples = sum(client_sizes)
    if total_examples == 0:
        return global_weights.copy()

    weighted_update = np.zeros_like(global_weights, dtype=np.float64)
    for update, n in zip(client_updates, client_sizes):
        weighted_update += (n / total_examples) * update
    return global_weights + weighted_update


def aggregate_client_deltas(
    client_deltas: list[np.ndarray], client_sizes: list[int]
) -> np.ndarray:
    """Aggregate weight deltas (not full weights) with sample weighting."""
    total = sum(client_sizes)
    if total == 0:
        return np.zeros_like(client_deltas[0]) if client_deltas else np.array([])
    agg = np.zeros_like(client_deltas[0], dtype=np.float64)
    for delta, n in zip(client_deltas, client_sizes):
        agg += (n / total) * delta
    return agg
