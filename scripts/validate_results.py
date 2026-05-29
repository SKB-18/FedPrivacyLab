"""
Validate experiment results against real-life expectations.

Exit 0 if all checks pass; prints human-readable report.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.analytics.federated_analytics import run_analytics_round
from app.config_loader import set_active_dataset
from app.coordinator.client_registry import clear_registry, populate_registry
from app.coordinator.experiment_runner import ExperimentRunner
from app.database import SessionLocal, init_db
from app.schemas import ExperimentStartRequest
from app.simulation.real_telemetry_loader import get_real_data_summary, load_real_hdfs_clients


def check_real_data_loaded() -> tuple[bool, str]:
    s = get_real_data_summary(partition_by="component")
    ok = (
        s["num_clients"] >= 3
        and s["total_events"] >= 500
        and s["failure_rate"] > 0
    )
    return ok, (
        f"clients={s['num_clients']} events={s['total_events']} "
        f"fail_rate={s['failure_rate']:.2%} partition={s.get('partition')}"
    )


def check_federated_matches_true_no_dp() -> tuple[bool, str]:
    set_active_dataset("real_hdfs_loghub")
    clients = load_real_hdfs_clients(max_clients=30)
    records_map = {c.client_id: [r.to_dict() for r in c.records] for c in clients}
    ids = list(records_map.keys())[:20]
    cohort = {cid: "cohort_a" for cid in ids}
    results, _ = run_analytics_round(
        records_map, ids, dp_enabled=False, cohort_map=cohort
    )
    failures = [r for r in results if r.metric_name == "average_latency_by_feature" and not r.feature]
    if not failures:
        return False, "no latency metric"
    r = failures[0]
    err = r.relative_error or 0
    ok = err < 0.15
    return ok, f"latency rel_err={err:.4f} true={r.true_value:.1f} fed={r.federated_value:.1f}"


def check_histogram_meta_consistent() -> tuple[bool, str]:
    set_active_dataset("real_hdfs_loghub")
    clients = load_real_hdfs_clients(partition_by="component")
    records_map = {c.client_id: [r.to_dict() for r in c.records] for c in clients}
    ids = list(records_map.keys())
    _, meta = run_analytics_round(records_map, ids, dp_enabled=False)
    th, h = meta.get("true_histogram"), meta.get("histogram")
    ok = th == h and sum(h.values()) > 0
    empty = meta.get("empty_buckets", [])
    return ok, f"events_in_round={meta.get('events_in_round')} empty_buckets={empty}"


def check_dp_increases_error() -> tuple[bool, str]:
    set_active_dataset("real_hdfs_loghub")
    clients = load_real_hdfs_clients(max_clients=30)
    records_map = {c.client_id: [r.to_dict() for r in c.records] for c in clients}
    ids = list(records_map.keys())[:20]
    cohort = {cid: "cohort_a" for cid in ids}
    _, _ = run_analytics_round(records_map, ids, dp_enabled=False, cohort_map=cohort)
    results_dp, _ = run_analytics_round(
        records_map, ids, dp_enabled=True, epsilon=0.5, cohort_map=cohort
    )
    fr = next(r for r in results_dp if r.metric_name == "count_failures_by_feature" and not r.feature)
    if fr.dp_noisy_value is None:
        return False, "no dp value"
    dp_err = abs(fr.dp_noisy_value - fr.true_value) / max(fr.true_value, 1e-6)
    fed_err = fr.relative_error or 0
    ok = dp_err >= fed_err * 0.5 or dp_err > 0.01
    return ok, f"fed_err={fed_err:.4f} dp_err={dp_err:.4f}"


def check_learning_pipeline() -> tuple[bool, str]:
    init_db()
    db = SessionLocal()
    set_active_dataset("real_hdfs_loghub")
    runner = ExperimentRunner(db)
    req = ExperimentStartRequest(
        name="validate_learning",
        mode="fedavg",
        dataset="real_hdfs_loghub",
        num_clients=5,
        rounds=3,
        clients_per_round=4,
        local_epochs=3,
        random_seed=42,
    )
    exp = runner.create_experiment(req)
    runner.initialize_clients(exp, req)
    runner.run_full_experiment(exp.id)
    from app.models import RoundMetric

    m = db.query(RoundMetric).filter(RoundMetric.experiment_id == exp.id).all()
    clear_registry(exp.id)
    db.close()
    if not m:
        return False, "no metrics"
    acc = m[-1].accuracy or 0
    ok = acc > 0.7
    return ok, f"final_accuracy={acc:.4f}"


def main():
    checks = [
        ("Real HDFS data loaded (subsystem clients)", check_real_data_loaded),
        ("Federated latency ~ true without DP", check_federated_matches_true_no_dp),
        ("Histogram meta true == federated", check_histogram_meta_consistent),
        ("DP adds measurable noise", check_dp_increases_error),
        ("Learning pipeline runs on real data", check_learning_pipeline),
    ]
    print("FedPrivacyLab Result Validation")
    print("=" * 50)
    all_ok = True
    for name, fn in checks:
        try:
            ok, detail = fn()
        except Exception as e:
            ok, detail = False, str(e)
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {name}: {detail}")
        all_ok = all_ok and ok
    print("=" * 50)
    print("OVERALL:", "PASS" if all_ok else "FAIL — see fixes above")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
