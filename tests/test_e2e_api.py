"""End-to-end API tests."""

import pytest
from fastapi.testclient import TestClient

from app.database import Base, engine
from app.main import app


@pytest.fixture
def client():
    Base.metadata.create_all(bind=engine)
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/api/v1/health")
    assert r.status_code == 200
    assert r.json()["status"] == "healthy"


def test_simulate_clients(client):
    r = client.post(
        "/api/v1/simulate/clients",
        json={"num_clients": 50, "min_records": 5, "max_records": 10, "random_seed": 1},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["num_clients"] == 50


def test_experiment_lifecycle(client):
    start = client.post(
        "/api/v1/experiments/start",
        json={
            "name": "e2e_analytics",
            "mode": "federated_analytics",
            "num_clients": 120,
            "rounds": 2,
            "clients_per_round": 100,
            "dp_enabled": True,
        },
    )
    assert start.status_code == 200
    exp_id = start.json()["id"]

    rnd = client.post(f"/api/v1/experiments/{exp_id}/run-round")
    assert rnd.status_code == 200

    analytics = client.get(f"/api/v1/experiments/{exp_id}/analytics")
    assert analytics.status_code == 200
    assert len(analytics.json()) > 0

    privacy = client.get(f"/api/v1/experiments/{exp_id}/privacy-report")
    assert privacy.status_code == 200
    assert privacy.json()["raw_data_centralized"] is False
