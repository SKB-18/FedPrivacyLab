"""L2 norm clipping for federated learning updates."""

import numpy as np


def clip_update(
    update_vector: np.ndarray, max_norm: float
) -> tuple[np.ndarray, bool]:
    """Clip update vector to max L2 norm. Returns (clipped_update, was_clipped)."""
    norm = float(np.linalg.norm(update_vector))
    if norm <= max_norm or norm == 0:
        return update_vector.copy(), False
    return update_vector * (max_norm / norm), True
