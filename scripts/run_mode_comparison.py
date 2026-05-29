"""
Run all five experiment modes with matched settings and export privacy–utility comparison.

Usage:
    python scripts/run_mode_comparison.py
    python scripts/run_mode_comparison.py --rounds 5 --clients 150
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from app.database import SessionLocal, init_db
from app.models import AnalyticsResult, Experiment, RoundMetric
from app.schemas import ExperimentStartRequest

OUTPUT_DIR = ROOT / "data" / "comparisons"


def reset_ml_runtime(seed: int) -> None:
    """Isolate TensorFlow/Keras and NumPy state between back-to-back experiments."""
    import random

    import numpy as np
    import tensorflow as tf

    tf.keras.backend.clear_session()
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

MODE_CONFIGS: list[dict] = [
    {
        "mode": "centralized_baseline",
        "name": "centralized_baseline",
        "dp_enabled": False,
        "secure_agg_enabled": False,
    },
    {
        "mode": "fedavg",
        "name": "fedavg_plain",
        "dp_enabled": False,
        "secure_agg_enabled": False,
    },
    {
        "mode": "fedavg_secureagg",
        "name": "fedavg_secureagg",
        "dp_enabled": False,
        "secure_agg_enabled": True,
    },
    {
        "mode": "fedavg_secureagg_dp",
        "name": "fedavg_secureagg_dp",
        "dp_enabled": True,
        "secure_agg_enabled": True,
        "noise_multiplier": 0.12,
        "epsilon": 2.0,
    },
    {
        "mode": "federated_analytics",
        "name": "federated_analytics_dp",
        "dp_enabled": True,
        "secure_agg_enabled": False,
    },
]


def build_request(
    mode_cfg: dict,
    *,
    num_clients: int,
    rounds: int,
    clients_per_round: int,
    local_epochs: int,
    seed: int,
    dataset: str = "synthetic_telemetry",
) -> ExperimentStartRequest:
    return ExperimentStartRequest(
        name=f"comparison_{mode_cfg['name']}",
        mode=mode_cfg["mode"],  # type: ignore[arg-type]
        dataset=dataset,
        num_clients=num_clients,
        rounds=rounds,
        clients_per_round=clients_per_round,
        local_epochs=local_epochs,
        dropout_rate=0.15,
        secure_agg_enabled=mode_cfg["secure_agg_enabled"],
        dp_enabled=mode_cfg["dp_enabled"],
        clipping_norm=1.0,
        noise_multiplier=float(mode_cfg.get("noise_multiplier", 0.5)),
        epsilon=float(mode_cfg.get("epsilon", 2.0)),
        delta=1e-6,
        random_seed=seed,
        non_iid_severity=1.0,
    )


def summarize_experiment(db, exp_id: int, mode: str) -> dict:
    metrics = (
        db.query(RoundMetric)
        .filter(RoundMetric.experiment_id == exp_id)
        .order_by(RoundMetric.round_number)
        .all()
    )
    analytics = (
        db.query(AnalyticsResult)
        .filter(AnalyticsResult.experiment_id == exp_id)
        .all()
    )

    row: dict = {
        "experiment_id": exp_id,
        "mode": mode,
        "final_accuracy": None,
        "final_roc_auc": None,
        "final_f1": None,
        "mean_relative_error": None,
        "rounds_completed": len(metrics),
    }

    if metrics:
        last = metrics[-1]
        row["final_accuracy"] = last.accuracy
        row["final_roc_auc"] = last.roc_auc
        row["final_f1"] = last.f1_score

    if analytics:
        errors = [a.relative_error for a in analytics if a.relative_error is not None and not a.suppressed]
        if errors:
            row["mean_relative_error"] = sum(errors) / len(errors)

    return row


def export_charts(summary_df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    learning = summary_df.dropna(subset=["final_accuracy"], how="all").copy()
    if not learning.empty:
        fig = px.bar(
            learning,
            x="mode",
            y="final_accuracy",
            color="mode",
            title="Final Accuracy by Experiment Mode",
            text_auto=".3f",
        )
        fig.update_layout(showlegend=False, yaxis_title="Accuracy")
        fig.write_html(str(out_dir / "accuracy_by_mode.html"))

        if learning["final_roc_auc"].notna().any():
            fig2 = px.bar(
                learning,
                x="mode",
                y="final_roc_auc",
                color="mode",
                title="Final ROC-AUC by Experiment Mode",
                text_auto=".3f",
            )
            fig2.update_layout(showlegend=False)
            fig2.write_html(str(out_dir / "roc_auc_by_mode.html"))

    analytics = summary_df.dropna(subset=["mean_relative_error"])
    if not analytics.empty:
        fig3 = px.bar(
            analytics,
            x="mode",
            y="mean_relative_error",
            color="mode",
            title="Mean Analytics Relative Error (lower = better utility)",
            text_auto=".3f",
        )
        fig3.update_layout(showlegend=False)
        fig3.write_html(str(out_dir / "analytics_error_by_mode.html"))

    # Combined privacy–utility scatter (learning modes)
    if not learning.empty:
        privacy_score = []
        for mode in learning["mode"]:
            if "dp" in mode or "analytics" in mode:
                privacy_score.append(0.8)
            elif "secureagg" in mode:
                privacy_score.append(0.6)
            elif mode == "fedavg_plain":
                privacy_score.append(0.3)
            else:
                privacy_score.append(0.1)
        learning = learning.copy()
        learning["privacy_strength"] = privacy_score
        fig4 = px.scatter(
            learning,
            x="privacy_strength",
            y="final_accuracy",
            color="mode",
            size_max=12,
            title="Privacy Strength vs Model Accuracy",
            labels={"privacy_strength": "Relative privacy (higher = stronger)"},
        )
        fig4.write_html(str(out_dir / "privacy_utility_tradeoff.html"))

    # Summary table figure
    display = summary_df.fillna("—")
    fig5 = go.Figure(
        data=[
            go.Table(
                header=dict(values=list(display.columns), fill_color="#3498db", font=dict(color="white")),
                cells=dict(values=[display[c].astype(str) for c in display.columns], fill_color="#ecf0f1"),
            )
        ]
    )
    fig5.update_layout(title="Mode Comparison Summary")
    fig5.write_html(str(out_dir / "summary_table.html"))


def run_comparison(
    num_clients: int = 150,
    rounds: int = 4,
    clients_per_round: int = 60,
    local_epochs: int = 2,
    seed: int = 42,
    dataset: str = "synthetic_telemetry",
) -> Path:
    init_db()
    summaries: list[dict] = []
    single_script = ROOT / "scripts" / "run_single_comparison_mode.py"

    print(f"FedPrivacyLab mode comparison — {len(MODE_CONFIGS)} modes, seed={seed}")
    print("(each mode in a fresh Python process for stable TensorFlow on Windows)")
    print("-" * 60)

    for mode_cfg in MODE_CONFIGS:
        mode = mode_cfg["mode"]
        print(f"Running {mode} ...", end=" ", flush=True)
        proc = subprocess.run(
            [
                sys.executable,
                str(single_script),
                "--mode",
                mode,
                "--clients",
                str(num_clients),
                "--rounds",
                str(rounds),
                "--clients-per-round",
                str(clients_per_round),
                "--local-epochs",
                str(local_epochs),
                "--seed",
                str(seed),
                "--dataset",
                dataset,
            ],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            print("FAILED")
            print(proc.stderr[-2000:] if proc.stderr else proc.stdout)
            raise RuntimeError(f"Mode {mode} failed (exit {proc.returncode})")
        line = proc.stdout.strip().splitlines()[-1]
        summary = json.loads(line)
        summaries.append(summary)
        acc = summary.get("final_accuracy")
        err = summary.get("mean_relative_error")
        detail = f"accuracy={acc:.4f}" if acc is not None else f"rel_err={err:.4f}" if err else "done"
        print(detail)

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = OUTPUT_DIR / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(summaries)
    df.to_csv(out_dir / "comparison_summary.csv", index=False)
    with open(out_dir / "comparison_summary.json", "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)

    export_charts(df, out_dir)

    print("-" * 60)
    print(f"Results saved to: {out_dir}")
    print("  - comparison_summary.csv")
    print("  - privacy_utility_tradeoff.html (open in browser)")
    return out_dir


def main():
    parser = argparse.ArgumentParser(description="Compare all FedPrivacyLab experiment modes")
    parser.add_argument("--clients", type=int, default=150, help="Number of simulated clients")
    parser.add_argument("--rounds", type=int, default=4, help="Rounds per learning experiment")
    parser.add_argument("--clients-per-round", type=int, default=60)
    parser.add_argument("--local-epochs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--dataset",
        type=str,
        default="synthetic_telemetry",
        choices=["synthetic_telemetry", "real_hdfs_loghub", "real_hdfs_loghub_pid"],
        help="real_hdfs_loghub = 5 subsystems; real_hdfs_loghub_pid = 13 process clients",
    )
    args = parser.parse_args()

    if args.clients < 120:
        print("Warning: federated_analytics needs >=120 clients for unsuppressed cohort reporting.")

    run_comparison(
        num_clients=args.clients,
        rounds=args.rounds,
        clients_per_round=args.clients_per_round,
        local_epochs=args.local_epochs,
        seed=args.seed,
        dataset=args.dataset,
    )


if __name__ == "__main__":
    main()
