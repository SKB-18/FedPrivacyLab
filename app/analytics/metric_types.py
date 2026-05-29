"""Metric metadata for correct visualization (units and chart grouping)."""

# Metrics that are rates in [0, 1]
RATE_METRICS = {
    "count_failures_by_feature",
    "click_rate_by_feature",
    "poor_experience_rate",
}

# Metrics in milliseconds
LATENCY_METRICS = {"average_latency_by_feature"}

# Count metrics (integer-like)
COUNT_METRICS = {"daily_active_clients_by_cohort"}

# Histogram quality score [0, 1] — not comparable to rates/latency on same axis
HISTOGRAM_METRICS = {"histogram_latency_by_bucket"}


def metric_display_name(name: str) -> str:
    labels = {
        "count_failures_by_feature": "Failure rate",
        "average_latency_by_feature": "Avg latency (ms)",
        "click_rate_by_feature": "Click / engagement rate",
        "poor_experience_rate": "Poor experience rate",
        "histogram_latency_by_bucket": "Histogram match score",
        "daily_active_clients_by_cohort": "Active clients (largest cohort)",
    }
    return labels.get(name, name)
