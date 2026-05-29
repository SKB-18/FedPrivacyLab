"""Quantile estimation from federated bounded summaries (approximate)."""

import numpy as np


def approximate_median_from_histogram(hist: dict[str, int]) -> float:
    """Approximate median latency from bucket histogram."""
    bucket_midpoints = {
        "0-500": 250,
        "500-1000": 750,
        "1000-2000": 1500,
        "2000-5000": 3500,
        "5000+": 5500,
    }
    total = sum(hist.values())
    if total == 0:
        return 0.0
    cumulative = 0
    target = total / 2
    for bucket in ["0-500", "500-1000", "1000-2000", "2000-5000", "5000+"]:
        count = hist.get(bucket, 0)
        cumulative += count
        if cumulative >= target:
            return bucket_midpoints.get(bucket, 0.0)
    return 0.0


def federated_quantile_from_local(
    local_values: list[float], quantile: float = 0.5
) -> float:
    """Centralized quantile from local aggregates (debug / baseline)."""
    if not local_values:
        return 0.0
    return float(np.quantile(local_values, quantile))
