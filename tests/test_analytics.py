"""Unit tests for federated analytics."""

from app.analytics.federated_analytics import compute_true_centralized_metrics, run_analytics_round
from app.analytics.local_summary import compute_local_summary


def test_local_summary():
    records = [
        {"feature": "writing_tools", "latency_ms": 1000, "failure": 1, "clicked_recommendation": 0},
        {"feature": "writing_tools", "latency_ms": 500, "failure": 0, "clicked_recommendation": 1},
    ]
    s = compute_local_summary(records, feature="writing_tools")
    assert s.failure_count == 1
    assert s.event_count == 2


def test_federated_analytics_round():
    records_map = {
        f"client_{i:03d}": [
            {
                "feature": "writing_tools",
                "latency_ms": 800 + i,
                "failure": i % 2,
                "clicked_recommendation": 1,
                "label_poor_experience": 0,
            }
            for _ in range(10)
        ]
        for i in range(120)
    }
    selected = list(records_map.keys())[:100]
    results, meta = run_analytics_round(
        records_map,
        selected,
        dp_enabled=False,
        round_number=1,
        cohort_map={c: "en_US_mid" for c in selected},
    )
    assert len(results) >= 6
    names = {r.metric_name for r in results}
    assert "histogram_latency_by_bucket" in names
    assert "daily_active_clients_by_cohort" in names
    assert meta["participating_clients"] == 100
    assert not results[0].suppressed
    hist = [r for r in results if r.metric_name == "histogram_latency_by_bucket"]
    assert hist and hist[0].federated_value == hist[0].true_value
    assert meta.get("true_histogram") == meta.get("histogram")


def test_histogram_bucket_order_and_percentages():
    from app.analytics.histogram import histogram_percentages, latency_bucket

    assert latency_bucket(499) == "0-500"
    assert latency_bucket(500) == "500-1000"
    assert latency_bucket(999) == "500-1000"
    assert latency_bucket(5000) == "5000+"
    pct = histogram_percentages({"0-500": 1, "5000+": 3})
    assert abs(pct["0-500"] - 25.0) < 0.01
    assert abs(pct["500-1000"] - 0.0) < 0.01


def test_true_centralized_metrics():
    records = [{"feature": "a", "latency_ms": 100, "failure": 1, "clicked_recommendation": 0, "label_poor_experience": 1}]
    m = compute_true_centralized_metrics(records)
    assert m["failure_rate"] == 1.0
