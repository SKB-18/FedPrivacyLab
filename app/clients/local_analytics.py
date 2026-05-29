"""Client-side federated analytics computation."""

from app.analytics.local_summary import LocalSummary, compute_local_summary


def run_local_analytics(
    records: list[dict],
    feature: str | None = None,
    round_id: int = 1,
) -> LocalSummary:
    return compute_local_summary(records, feature=feature, round_id=round_id)
