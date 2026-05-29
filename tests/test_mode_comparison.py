"""Smoke test for mode comparison script (single fast mode)."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.run_mode_comparison import build_request, summarize_experiment
from app.coordinator.client_registry import clear_registry
from app.coordinator.experiment_runner import ExperimentRunner
from app.schemas import ExperimentStartRequest


def test_build_request_valid():
    req = build_request(
        {"mode": "fedavg", "name": "fedavg", "dp_enabled": False, "secure_agg_enabled": False},
        num_clients=50,
        rounds=2,
        clients_per_round=20,
        local_epochs=1,
        seed=1,
    )
    assert req.mode == "fedavg"
    assert req.num_clients == 50


def test_single_mode_comparison_run(db_session):
    runner = ExperimentRunner(db_session)
    req = ExperimentStartRequest(
        name="smoke_fedavg",
        mode="fedavg",
        num_clients=40,
        rounds=1,
        clients_per_round=20,
        local_epochs=1,
    )
    exp = runner.create_experiment(req)
    runner.run_full_experiment(exp.id, config=req)
    summary = summarize_experiment(db_session, exp.id, "fedavg")
    assert summary["rounds_completed"] >= 1
    clear_registry(exp.id)
