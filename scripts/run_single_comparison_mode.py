"""Run one comparison mode in an isolated process (clean TensorFlow state on Windows)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.coordinator.client_registry import clear_registry
from app.coordinator.experiment_runner import ExperimentRunner
from app.database import SessionLocal, init_db
from scripts.run_mode_comparison import MODE_CONFIGS, build_request, summarize_experiment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=[m["mode"] for m in MODE_CONFIGS])
    parser.add_argument("--clients", type=int, default=200)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--clients-per-round", type=int, default=80)
    parser.add_argument("--local-epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset", default="synthetic_telemetry")
    args = parser.parse_args()

    mode_cfg = next(m for m in MODE_CONFIGS if m["mode"] == args.mode)
    req = build_request(
        mode_cfg,
        num_clients=args.clients,
        rounds=args.rounds,
        clients_per_round=min(args.clients_per_round, args.clients),
        local_epochs=args.local_epochs,
        seed=args.seed,
        dataset=args.dataset,
    )

    init_db()
    db = SessionLocal()
    try:
        runner = ExperimentRunner(db)
        exp = runner.create_experiment(req)
        runner.initialize_clients(exp, req)
        runner.run_full_experiment(exp.id, config=req)
        summary = summarize_experiment(db, exp.id, args.mode)
        print(json.dumps(summary))
        clear_registry(exp.id)
    finally:
        db.close()


if __name__ == "__main__":
    main()
