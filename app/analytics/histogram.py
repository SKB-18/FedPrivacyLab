"""Histogram aggregation for federated analytics."""

from typing import Any

LATENCY_BUCKET_ORDER = ["0-500", "500-1000", "1000-2000", "2000-5000", "5000+"]


def latency_bucket(latency_ms: float) -> str:
    """Map latency (ms) to spec histogram bucket."""
    if latency_ms < 500:
        return "0-500"
    if latency_ms < 1000:
        return "500-1000"
    if latency_ms < 2000:
        return "1000-2000"
    if latency_ms < 5000:
        return "2000-5000"
    return "5000+"


def aggregate_histograms(summaries: list[dict]) -> dict[str, int]:
    """Merge latency bucket histograms from local summaries."""
    global_hist: dict[str, int] = {}
    for s in summaries:
        buckets = s.get("latency_buckets") or {}
        for bucket, count in buckets.items():
            global_hist[bucket] = global_hist.get(bucket, 0) + int(count)
    return global_hist


def histogram_to_series(hist: dict[str, int]) -> list[dict[str, Any]]:
    return [
        {"bucket": b, "count": hist.get(b, 0)}
        for b in LATENCY_BUCKET_ORDER
    ]


def histogram_percentages(hist: dict[str, int]) -> dict[str, float]:
    total = sum(hist.get(b, 0) for b in LATENCY_BUCKET_ORDER)
    if total == 0:
        return {b: 0.0 for b in LATENCY_BUCKET_ORDER}
    return {b: 100.0 * hist.get(b, 0) / total for b in LATENCY_BUCKET_ORDER}
