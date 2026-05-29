"""Integration tests: coordinator + DB + simulation."""

import numpy as np

from app.coordinator.client_registry import clear_registry, populate_registry
from app.coordinator.experiment_runner import ExperimentRunner
from app.schemas import ExperimentStartRequest


def test_analytics_experiment_round(db_session):
    config = ExperimentStartRequest(
        name="integration_analytics",
        mode="federated_analytics",
        num_clients=120,
        rounds=2,
        clients_per_round=100,
        dp_enabled=True,
        epsilon=2.0,
    )
    runner = ExperimentRunner(db_session)
    exp = runner.create_experiment(config)
    runner.initialize_clients(exp, config)
    result = runner.run_single_round(exp.id)
    assert result["round"] == 1
    assert result["completed"] > 0
    clear_registry(exp.id)


def test_learning_experiment_round(db_session):
    config = ExperimentStartRequest(
        name="integration_fedavg",
        mode="fedavg_secureagg_dp",
        num_clients=50,
        rounds=3,
        clients_per_round=20,
        local_epochs=1,
        dp_enabled=True,
        secure_agg_enabled=True,
    )
    runner = ExperimentRunner(db_session)
    exp = runner.create_experiment(config)
    runner.initialize_clients(exp, config)
    result = runner.run_single_round(exp.id)
    assert "metrics" in result
    assert result["completed"] >= 0
    clear_registry(exp.id)


def test_synthetic_generation():
    from app.simulation.synthetic_telemetry import generate_synthetic_clients

    clients = generate_synthetic_clients(num_clients=20, seed=1)
    assert len(clients) == 20
    assert all(len(c.records) >= 5 for c in clients)
