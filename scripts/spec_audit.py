"""Automated cross-check against FedPrivacyLab_Architecture_Spec.pdf structure."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

REQUIRED_PATHS = [
    "app/main.py",
    "app/database.py",
    "app/models.py",
    "app/schemas.py",
    "app/api/experiment_routes.py",
    "app/api/analytics_routes.py",
    "app/api/learning_routes.py",
    "app/api/privacy_routes.py",
    "app/api/metrics_routes.py",
    "app/coordinator/round_coordinator.py",
    "app/coordinator/client_registry.py",
    "app/coordinator/experiment_runner.py",
    "app/clients/client.py",
    "app/clients/telemetry_client.py",
    "app/clients/emnist_client.py",
    "app/clients/local_analytics.py",
    "app/clients/local_training.py",
    "app/analytics/federated_analytics.py",
    "app/analytics/local_summary.py",
    "app/analytics/histogram.py",
    "app/analytics/quantiles.py",
    "app/learning/model.py",
    "app/learning/fedavg.py",
    "app/learning/centralized_baseline.py",
    "app/learning/evaluator.py",
    "app/privacy/contribution_bounding.py",
    "app/privacy/update_clipping.py",
    "app/privacy/secure_aggregation.py",
    "app/privacy/dp_mechanisms.py",
    "app/privacy/privacy_accountant.py",
    "app/privacy/risk_report.py",
    "app/simulation/synthetic_telemetry.py",
    "app/simulation/real_telemetry_loader.py",
    "app/simulation/non_iid_partition.py",
    "app/simulation/dropout_simulator.py",
    "app/simulation/attack_simulator.py",
    "dashboard/streamlit_app.py",
    "config/experiment_config.yaml",
    "config/privacy_config.yaml",
    "tests/test_fedavg.py",
    "tests/test_contribution_bounding.py",
    "tests/test_update_clipping.py",
    "tests/test_secure_aggregation.py",
    "tests/test_dp_mechanisms.py",
    "tests/test_non_iid_partition.py",
    "README.md",
    "docker-compose.yml",
]

REQUIRED_ANALYTICS = [
    "count_failures_by_feature",
    "average_latency_by_feature",
    "click_rate_by_feature",
    "histogram_latency_by_bucket",
    "daily_active_clients_by_cohort",
    "poor_experience_rate",
]

MODES = [
    "centralized_baseline",
    "federated_analytics",
    "fedavg",
    "fedavg_secureagg",
    "fedavg_secureagg_dp",
]


def run_audit() -> dict:
    missing = [p for p in REQUIRED_PATHS if not (ROOT / p).exists()]
    from app.analytics.federated_analytics import ANALYTICS_QUERIES

    analytics_ok = all(q in ANALYTICS_QUERIES for q in REQUIRED_ANALYTICS)
    tables = ["experiments", "round_metrics", "analytics_results", "privacy_reports", "client_participation"]
    from app.database import Base, engine
    from app import models  # noqa: F401

    Base.metadata.create_all(bind=engine)
    db_tables = set(Base.metadata.tables.keys())

    return {
        "files_missing": missing,
        "files_ok": len(REQUIRED_PATHS) - len(missing),
        "files_total": len(REQUIRED_PATHS),
        "analytics_queries_ok": analytics_ok,
        "analytics_queries": ANALYTICS_QUERIES,
        "experiment_modes": MODES,
        "db_tables_ok": all(t in db_tables for t in tables),
        "complete": len(missing) == 0 and analytics_ok and all(t in db_tables for t in tables),
    }


if __name__ == "__main__":
    r = run_audit()
    print("=" * 60)
    print("FedPrivacyLab Spec Compliance Audit")
    print("=" * 60)
    print(f"Files: {r['files_ok']}/{r['files_total']}")
    print(f"Analytics queries (6/6): {r['analytics_queries_ok']}")
    print(f"DB tables: {r['db_tables_ok']}")
    print(f"OVERALL COMPLETE: {r['complete']}")
    if r["files_missing"]:
        print("Missing:", r["files_missing"])
    sys.exit(0 if r["complete"] else 1)
