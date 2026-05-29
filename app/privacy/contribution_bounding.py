"""Contribution bounding for federated analytics."""

from dataclasses import dataclass

from app.config_loader import get_privacy_bounds


@dataclass
class BoundedSummary:
    failure_count: int
    event_count: int
    latency_sum: float
    latency_count: int
    click_count: int
    bounded: bool = False


def get_bounds() -> dict:
    return get_privacy_bounds()


def bound_analytics_summary(
    failure_count: int,
    event_count: int,
    latency_sum: float,
    latency_count: int,
    click_count: int,
    bounds: dict | None = None,
) -> BoundedSummary:
    """Bound per-client analytics contributions before aggregation."""
    b = bounds or get_bounds()
    max_events = int(b.get("max_events_per_client_per_round", 20))
    max_failures = int(b.get("max_failures_per_client_per_round", 5))
    max_latency_sum = float(b.get("max_latency_sum_per_client", 50000))

    bounded_events = min(event_count, max_events)
    bounded_failures = min(failure_count, max_failures)
    bounded_latency_sum = min(latency_sum, max_latency_sum)
    bounded_latency_count = min(latency_count, bounded_events)

    was_bounded = (
        event_count > max_events
        or failure_count > max_failures
        or latency_sum > max_latency_sum
    )

    return BoundedSummary(
        failure_count=bounded_failures,
        event_count=bounded_events,
        latency_sum=bounded_latency_sum,
        latency_count=bounded_latency_count,
        click_count=min(click_count, bounded_events),
        bounded=was_bounded,
    )
