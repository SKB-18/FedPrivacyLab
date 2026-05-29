"""Run a quick demo experiment via the API."""

import json
import sys
import time

import httpx

API = "http://127.0.0.1:8000"


def main():
    payload = {
        "name": "demo_fedavg_dp",
        "mode": "fedavg_secureagg_dp",
        "num_clients": 80,
        "rounds": 3,
        "clients_per_round": 30,
        "local_epochs": 1,
        "dropout_rate": 0.1,
        "secure_agg_enabled": True,
        "dp_enabled": True,
        "epsilon": 2.0,
    }
    with httpx.Client(base_url=API, timeout=300.0) as client:
        health = client.get("/api/v1/health")
        health.raise_for_status()
        start = client.post("/api/v1/experiments/start", json=payload)
        start.raise_for_status()
        exp_id = start.json()["id"]
        print(f"Started experiment {exp_id}")

        for _ in range(payload["rounds"]):
            r = client.post(f"/api/v1/experiments/{exp_id}/run-round")
            r.raise_for_status()
            print(f"Round {r.json()['round_number']} done")
            time.sleep(0.5)

        metrics = client.get(f"/api/v1/experiments/{exp_id}/metrics")
        privacy = client.get(f"/api/v1/experiments/{exp_id}/privacy-report")
        print("Metrics:", json.dumps(metrics.json(), indent=2))
        print("Privacy:", json.dumps(privacy.json(), indent=2))


if __name__ == "__main__":
    try:
        main()
    except httpx.ConnectError:
        print("Start API first: uvicorn app.main:app --reload", file=sys.stderr)
        sys.exit(1)
