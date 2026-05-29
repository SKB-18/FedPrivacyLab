"""Validation tests for fair metrics and real data scale."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config_loader import get_cohort_threshold, get_privacy_bounds, set_active_dataset
from app.simulation.real_telemetry_loader import get_real_data_summary


def test_real_hdfs_has_enough_clients():
    set_active_dataset("real_hdfs_loghub")
    s = get_real_data_summary()
    assert s["num_clients"] >= 3, f"expected 3+ subsystem clients, got {s['num_clients']}"
    assert s["failure_rate"] > 0, "real HDFS WARN levels should yield non-zero failure rate"


def test_real_hdfs_bounds_override():
    set_active_dataset("real_hdfs_loghub")
    b = get_privacy_bounds()
    assert b["max_events_per_client_per_round"] >= 50
    assert get_cohort_threshold() == 5


def test_fair_latency_comparison():
    from scripts.validate_results import check_federated_matches_true_no_dp

    set_active_dataset("real_hdfs_loghub")
    ok, detail = check_federated_matches_true_no_dp()
    assert ok, detail
