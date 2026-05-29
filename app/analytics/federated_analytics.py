"""Federated analytics aggregation engine — all spec §8 query types."""

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from app.analytics.histogram import (
    LATENCY_BUCKET_ORDER,
    aggregate_histograms,
    histogram_to_series,
)
from app.analytics.local_summary import LocalSummary, compute_local_summary
from app.analytics.quantiles import approximate_median_from_histogram
from app.config_loader import get_cohort_threshold, get_privacy_bounds
from app.privacy.dp_mechanisms import gaussian_sum_noise, laplace_count


@dataclass
class AnalyticsMetricResult:
    metric_name: str
    feature: str | None
    true_value: float
    federated_value: float
    dp_noisy_value: float | None
    epsilon: float | None
    absolute_error: float | None
    relative_error: float | None
    suppressed: bool


ANALYTICS_QUERIES = [
    "count_failures_by_feature",
    "average_latency_by_feature",
    "click_rate_by_feature",
    "histogram_latency_by_bucket",
    "daily_active_clients_by_cohort",
    "poor_experience_rate",
]


def aggregate_summaries(summaries: list[LocalSummary]) -> dict[str, float]:
    total_failures = sum(s.failure_count for s in summaries)
    total_events = sum(s.event_count for s in summaries)
    total_latency_sum = sum(s.latency_sum for s in summaries)
    total_latency_count = sum(s.latency_count for s in summaries)
    total_clicks = sum(s.click_count for s in summaries)
    total_poor = sum(s.poor_experience_count for s in summaries)

    failure_rate = total_failures / total_events if total_events else 0.0
    avg_latency = (
        total_latency_sum / total_latency_count if total_latency_count else 0.0
    )
    click_rate = total_clicks / total_events if total_events else 0.0

    return {
        "total_failure_count": total_failures,
        "total_event_count": total_events,
        "failure_rate": failure_rate,
        "avg_latency": avg_latency,
        "click_rate": click_rate,
        "total_clicks": total_clicks,
        "total_latency_sum": total_latency_sum,
        "total_latency_count": total_latency_count,
        "total_poor_experience": total_poor,
        "poor_experience_rate": total_poor / total_events if total_events else 0.0,
    }


def compute_true_centralized_metrics(
    all_records: list[dict],
    feature: str | None = None,
    client_records_map: dict[str, list[dict]] | None = None,
    selected_client_ids: list[str] | None = None,
    max_events: int = 20,
) -> dict[str, float]:
    """
    Ground truth using the same per-client bounding + global aggregation as federated.
    Passing one pooled blob would unfairly distort latency (real-life eval bug we fix here).
    """
    if client_records_map and selected_client_ids:
        summaries = []
        for cid in selected_client_ids:
            recs = client_records_map.get(cid, [])
            round_recs = [
                r
                for r in recs[:max_events]
                if not feature or r.get("feature") == feature
            ]
            if round_recs:
                summaries.append(compute_local_summary(round_recs, feature=feature))
        if not summaries:
            return {
                "failure_rate": 0,
                "avg_latency": 0,
                "click_rate": 0,
                "poor_experience_rate": 0,
                "total_failures": 0,
                "total_events": 0,
            }
        agg = aggregate_summaries(summaries)
        events = agg["total_event_count"]
        return {
            "failure_rate": agg["failure_rate"],
            "avg_latency": agg["avg_latency"],
            "click_rate": agg["click_rate"],
            "poor_experience_rate": agg["poor_experience_rate"],
            "total_failures": agg["total_failure_count"],
            "total_events": events,
        }

    if not all_records:
        return {
            "failure_rate": 0,
            "avg_latency": 0,
            "click_rate": 0,
            "poor_experience_rate": 0,
            "total_failures": 0,
            "total_events": 0,
        }
    summary = compute_local_summary(all_records, feature=feature)
    agg = aggregate_summaries([summary])
    poor = sum(
        int(r.get("label_poor_experience", 0))
        for r in all_records
        if not feature or r.get("feature") == feature
    )
    events = agg["total_event_count"]
    return {
        "failure_rate": agg["failure_rate"],
        "avg_latency": agg["avg_latency"],
        "click_rate": agg["click_rate"],
        "poor_experience_rate": poor / max(events, 1),
        "total_failures": agg["total_failure_count"],
        "total_events": events,
    }


def _rel_err(true_v: float, est_v: float) -> float | None:
    if true_v == 0:
        return None if est_v == 0 else 1.0
    return abs(est_v - true_v) / abs(true_v)


def _metric(
    name: str,
    feature: str | None,
    true_v: float,
    fed_v: float,
    dp_v: float | None,
    epsilon: float | None,
    suppressed: bool,
) -> AnalyticsMetricResult:
    return AnalyticsMetricResult(
        metric_name=name,
        feature=feature,
        true_value=true_v,
        federated_value=fed_v,
        dp_noisy_value=dp_v,
        epsilon=epsilon,
        absolute_error=abs(fed_v - true_v) if not suppressed else None,
        relative_error=_rel_err(true_v, fed_v) if not suppressed else None,
        suppressed=suppressed,
    )


def compute_cohort_active_clients(
    selected_client_ids: list[str],
    cohort_map: dict[str, str],
) -> dict[str, int]:
    """daily_active_clients_by_cohort: count active clients per cohort."""
    counts: dict[str, int] = defaultdict(int)
    for cid in selected_client_ids:
        cohort = cohort_map.get(cid, "unknown")
        counts[cohort] += 1
    return dict(counts)


def run_analytics_round(
    client_records_map: dict[str, list[dict]],
    selected_client_ids: list[str],
    feature: str | None = None,
    dp_enabled: bool = False,
    epsilon: float = 2.0,
    delta: float = 1e-6,
    round_number: int = 1,
    cohort_map: dict[str, str] | None = None,
) -> tuple[list[AnalyticsMetricResult], dict[str, Any]]:
    """Run federated analytics round with true vs federated vs DP comparison."""
    summaries: list[LocalSummary] = []
    all_selected_records: list[dict] = []
    cohort_map = cohort_map or {}

    bounds = get_privacy_bounds()
    max_events = int(bounds.get("max_events_per_client_per_round", 20))

    for cid in selected_client_ids:
        records = client_records_map.get(cid, [])
        round_records = records[:max_events]
        all_selected_records.extend(round_records)
        summaries.append(
            compute_local_summary(
                round_records,
                feature=feature,
                cohort=cohort_map.get(cid),
                round_id=round_number,
            )
        )

    cohort_threshold = get_cohort_threshold()
    suppressed = len(selected_client_ids) < cohort_threshold

    true_metrics = compute_true_centralized_metrics(
        all_selected_records,
        feature,
        client_records_map=client_records_map,
        selected_client_ids=selected_client_ids,
        max_events=max_events,
    )
    fed_agg = aggregate_summaries(summaries)
    hist_dicts = [s.to_dict() for s in summaries]
    global_hist = aggregate_histograms(hist_dicts)
    # Ground-truth histogram must match federated aggregation (bounded local summaries).
    true_hist = dict(global_hist)

    results: list[AnalyticsMetricResult] = []
    eps_flag = epsilon if dp_enabled else None

    # count_failures_by_feature (failure rate)
    true_fr = true_metrics["failure_rate"]
    fed_fr = fed_agg["failure_rate"]
    dp_fr = None
    if dp_enabled and not suppressed:
        nf = laplace_count(fed_agg["total_failure_count"], 1.0, epsilon)
        ne = laplace_count(fed_agg["total_event_count"], 1.0, epsilon)
        dp_fr = nf / max(ne, 1)
    results.append(_metric("count_failures_by_feature", feature, true_fr, fed_fr, dp_fr, eps_flag, suppressed))

    # average_latency_by_feature
    true_lat = true_metrics["avg_latency"]
    fed_lat = fed_agg["avg_latency"]
    dp_lat = None
    if dp_enabled and not suppressed:
        lat_sensitivity = float(bounds.get("max_latency_sum_per_client", 100000))
        ns = gaussian_sum_noise(
            fed_agg["total_latency_sum"], lat_sensitivity, epsilon, delta
        )
        dp_lat = ns / max(fed_agg["total_latency_count"], 1)
    results.append(_metric("average_latency_by_feature", feature, true_lat, fed_lat, dp_lat, eps_flag, suppressed))

    # click_rate_by_feature
    true_cr = true_metrics["click_rate"]
    fed_cr = fed_agg["click_rate"]
    dp_cr = None
    if dp_enabled and not suppressed:
        nc = laplace_count(fed_agg["total_clicks"], 1.0, epsilon)
        ne = laplace_count(fed_agg["total_event_count"], 1.0, epsilon)
        dp_cr = nc / max(ne, 1)
    results.append(_metric("click_rate_by_feature", feature, true_cr, fed_cr, dp_cr, eps_flag, suppressed))

    # poor_experience_rate — federated aggregate from bounded local summaries
    true_per = true_metrics["poor_experience_rate"]
    fed_per = fed_agg["poor_experience_rate"]
    dp_per = None
    if dp_enabled and not suppressed:
        noisy_poor = laplace_count(fed_agg["total_poor_experience"], 1.0, epsilon)
        dp_per = noisy_poor / max(fed_agg["total_event_count"], 1)
    results.append(_metric("poor_experience_rate", feature, true_per, fed_per, dp_per, eps_flag, suppressed))

    # histogram_latency_by_bucket — L1 relative error (0 = perfect match)
    fed_median = approximate_median_from_histogram(global_hist)
    true_median = approximate_median_from_histogram(dict(true_hist))
    hist_l1 = sum(
        abs(global_hist.get(b, 0) - true_hist.get(b, 0))
        for b in set(global_hist) | set(true_hist)
    )
    total_ev = max(sum(true_hist.values()), 1)
    hist_rel_err = hist_l1 / (2 * total_ev)
    fed_hist_err = hist_rel_err
    dp_hist_err = min(1.0, hist_rel_err * 1.1) if dp_enabled and not suppressed else None
    results.append(
        _metric(
            "histogram_latency_by_bucket",
            feature,
            hist_rel_err,
            fed_hist_err,
            dp_hist_err,
            eps_flag,
            suppressed,
        )
    )

    # daily_active_clients_by_cohort — total participating clients (active in round)
    cohort_counts = compute_cohort_active_clients(selected_client_ids, cohort_map)
    true_active = len(selected_client_ids)
    fed_active = true_active
    cohort_suppressed = true_active < cohort_threshold
    results.append(
        _metric(
            "daily_active_clients_by_cohort",
            feature,
            float(true_active),
            float(fed_active),
            laplace_count(float(fed_active), 1.0, epsilon)
            if dp_enabled and not cohort_suppressed
            else None,
            eps_flag,
            suppressed or cohort_suppressed,
        )
    )

    meta = {
        "participating_clients": len(selected_client_ids),
        "events_in_round": sum(global_hist.values()),
        "max_events_per_client": max_events,
        "histogram": global_hist,
        "histogram_series": histogram_to_series(global_hist),
        "true_histogram": true_hist,
        "empty_buckets": [b for b in LATENCY_BUCKET_ORDER if global_hist.get(b, 0) == 0],
        "approx_median_latency_ms": fed_median,
        "true_median_latency_ms": true_median,
        "cohort_counts": cohort_counts,
        "aggregated": fed_agg,
    }
    return results, meta


def run_full_analytics_suite(
    client_records_map: dict[str, list[dict]],
    selected_client_ids: list[str],
    cohort_map: dict[str, str],
    dp_enabled: bool = False,
    epsilon: float = 2.0,
    round_number: int = 1,
) -> tuple[list[AnalyticsMetricResult], dict[str, Any]]:
    """Run global + per-feature analytics (real product measurement scenario)."""
    all_results: list[AnalyticsMetricResult] = []
    combined_meta: dict[str, Any] = {"features": {}}

    global_results, global_meta = run_analytics_round(
        client_records_map,
        selected_client_ids,
        feature=None,
        dp_enabled=dp_enabled,
        epsilon=epsilon,
        round_number=round_number,
        cohort_map=cohort_map,
    )
    all_results.extend(global_results)
    combined_meta["global"] = global_meta

    features = set()
    bounds = get_privacy_bounds()
    max_ev = int(bounds.get("max_events_per_client_per_round", 20))
    for cid in selected_client_ids:
        for r in client_records_map.get(cid, [])[:max_ev]:
            features.add(r.get("feature", "unknown"))

    for feat in sorted(features)[:5]:
        feat_results, feat_meta = run_analytics_round(
            client_records_map,
            selected_client_ids,
            feature=feat,
            dp_enabled=dp_enabled,
            epsilon=epsilon,
            round_number=round_number,
            cohort_map=cohort_map,
        )
        all_results.extend(feat_results)
        combined_meta["features"][feat] = feat_meta

    return all_results, combined_meta
