"""Simulate client dropout in federated rounds."""

import numpy as np


def apply_dropout(
    selected_client_ids: list[str],
    dropout_rate: float,
    seed: int | None = None,
) -> tuple[list[str], list[str]]:
    """
    Split selected clients into completed and dropped.
    Returns (completed_ids, dropped_ids).
    """
    if dropout_rate <= 0 or not selected_client_ids:
        return list(selected_client_ids), []

    rng = np.random.default_rng(seed)
    n_drop = int(len(selected_client_ids) * dropout_rate)
    if n_drop == 0:
        return list(selected_client_ids), []

    indices = rng.permutation(len(selected_client_ids))
    drop_set = set(indices[:n_drop])
    completed = []
    dropped = []
    for i, cid in enumerate(selected_client_ids):
        if i in drop_set:
            dropped.append(cid)
        else:
            completed.append(cid)
    return completed, dropped
