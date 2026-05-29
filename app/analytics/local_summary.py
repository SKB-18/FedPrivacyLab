"""Local client summary computation for federated analytics."""

from dataclasses import dataclass, field
from typing import Any

from app.analytics.histogram import latency_bucket
from app.config_loader import get_privacy_bounds
from app.privacy.contribution_bounding import bound_analytics_summary


@dataclass
class LocalSummary:
    feature: str | None = None
    failure_count: int = 0
    event_count: int = 0
    latency_sum: float = 0.0
    latency_count: int = 0
    click_count: int = 0
    poor_experience_count: int = 0
    latency_buckets: dict[str, int] = field(default_factory=dict)
    cohort: str | None = None
    active: bool = True

    def to_dict(self, round_id: int = 1) -> dict[str, Any]:
        return {
            "client_id_removed_before_storage": True,
            "round_id": round_id,
            "feature": self.feature,
            "failure_count": self.failure_count,
            "event_count": self.event_count,
            "latency_sum_clipped": self.latency_sum,
            "latency_count": self.latency_count,
            "click_count": self.click_count,
            "latency_buckets": self.latency_buckets,
            "cohort": self.cohort,
        }


def _clip_event_latency(latency_ms: float, bounds: dict) -> float:
    max_lat = float(bounds.get("max_latency_ms_per_event", 5000))
    return min(latency_ms, max_lat)


def compute_local_summary(
    records: list[dict],
    feature: str | None = None,
    cohort: str | None = None,
    round_id: int = 1,
    bounds: dict | None = None,
) -> LocalSummary:
    """Compute local summary from client records for one round."""
    bounds = bounds or get_privacy_bounds()
    summary = LocalSummary(feature=feature, cohort=cohort)

    for rec in records:
        if feature and rec.get("feature") != feature:
            continue
        lat = _clip_event_latency(float(rec.get("latency_ms", 0)), bounds)
        summary.event_count += 1
        summary.failure_count += int(rec.get("failure", 0))
        summary.latency_sum += lat
        summary.latency_count += 1
        summary.click_count += int(rec.get("clicked_recommendation", 0))
        summary.poor_experience_count += int(rec.get("label_poor_experience", 0))

        bucket = latency_bucket(lat)
        summary.latency_buckets[bucket] = summary.latency_buckets.get(bucket, 0) + 1

    bounded = bound_analytics_summary(
        summary.failure_count,
        summary.event_count,
        summary.latency_sum,
        summary.latency_count,
        summary.click_count,
    )
    summary.failure_count = bounded.failure_count
    summary.event_count = bounded.event_count
    summary.latency_sum = bounded.latency_sum
    summary.latency_count = bounded.latency_count
    summary.click_count = bounded.click_count
    # Poor experience bounded by event cap (cannot exceed events in round)
    summary.poor_experience_count = min(
        summary.poor_experience_count, summary.event_count
    )
    return summary
