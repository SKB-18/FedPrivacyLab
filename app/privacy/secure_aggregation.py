"""Secure aggregation simulation via additive masks."""

import numpy as np


def mask_update(update: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Add random mask to client update. Masks cancel when summed."""
    rng = np.random.default_rng(seed)
    mask = rng.normal(0, 1, size=update.shape)
    return update + mask, mask


def aggregate_masked_updates(
    masked_updates: list[np.ndarray], masks: list[np.ndarray]
) -> np.ndarray:
    """Sum masked updates and subtract masks to recover true aggregate."""
    if not masked_updates:
        raise ValueError("No masked updates to aggregate")
    return np.sum(masked_updates, axis=0) - np.sum(masks, axis=0)


def secure_aggregate_updates(
    updates: list[np.ndarray], base_seed: int = 42
) -> np.ndarray:
    """Mask each update, sum, and unmask — server sees aggregate only."""
    masked = []
    masks = []
    for i, update in enumerate(updates):
        m_update, mask = mask_update(update, base_seed + i)
        masked.append(m_update)
        masks.append(mask)
    return aggregate_masked_updates(masked, masks)
