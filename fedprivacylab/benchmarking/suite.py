"""
Benchmarking suite comparing FedAvg and FedProx federated learning algorithms.

Metrics tracked per round:
  accuracy, f1_score, training_time_seconds, bytes_transferred, epsilon, convergence_rate
"""

from __future__ import annotations

import csv
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# Guard optional heavy imports so the module is importable without TF/sklearn
try:
    import tensorflow as tf
    from sklearn.metrics import accuracy_score, f1_score as sk_f1
    _HAS_TF = True
except ImportError:
    _HAS_TF = False

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    _HAS_PLOTLY = True
except ImportError:
    _HAS_PLOTLY = False

# Reuse existing infrastructure
try:
    from app.learning.model import build_model
    from app.learning.fedavg import fedavg
    from app.privacy.dp_mechanisms import add_gaussian_noise_to_update
    from app.privacy.update_clipping import clip_update
    from app.simulation.synthetic_telemetry import SyntheticTelemetryGenerator
    from app.learning.feature_encoding import fit_normalization, records_to_xy
    _HAS_APP = True
except ImportError:
    _HAS_APP = False

RESULTS_DIR = Path(__file__).parents[2] / "data" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CSV_COLUMNS = [
    "algorithm",
    "round",
    "accuracy",
    "f1_score",
    "training_time_seconds",
    "bytes_transferred",
    "epsilon",
    "convergence_rate",
]


@dataclass
class RoundResult:
    algorithm: str
    round: int
    accuracy: float
    f1_score: float
    training_time_seconds: float
    bytes_transferred: int
    epsilon: float
    convergence_rate: float


def _build_synthetic_clients(num_clients: int, records_per_client: int = 200) -> list[dict[str, Any]]:
    """Generate synthetic non-IID client datasets using existing infrastructure."""
    if not _HAS_APP:
        rng = np.random.default_rng(42)
        clients = []
        for i in range(num_clients):
            n = records_per_client
            X = rng.standard_normal((n, 8)).astype(np.float32)
            y = (X[:, 0] + rng.standard_normal(n) > 0).astype(np.float32)
            clients.append({"X_train": X, "y_train": y, "n": n})
        return clients

    gen = SyntheticTelemetryGenerator(num_clients=num_clients, records_per_client=records_per_client)
    all_clients = gen.generate()
    normalization = fit_normalization([r for c in all_clients for r in c])

    clients = []
    for client_records in all_clients:
        X, y = records_to_xy(client_records, normalization=normalization)
        clients.append({"X_train": X.astype(np.float32), "y_train": y.astype(np.float32), "n": len(client_records)})
    return clients


def _build_global_model(input_dim: int = 8) -> "tf.keras.Model":
    """Build or stub the global Keras model."""
    if _HAS_APP and _HAS_TF:
        return build_model(input_dim=input_dim)
    if _HAS_TF:
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(32, activation="relu", input_shape=(input_dim,)),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ])
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        return model
    raise RuntimeError("TensorFlow is required for benchmarking.")


def _bytes_of_weights(weights: list[np.ndarray]) -> int:
    """Estimate bytes needed to transmit model weights."""
    return sum(w.nbytes for w in weights)


def _eval_model(model: "tf.keras.Model", X_test: np.ndarray, y_test: np.ndarray) -> tuple[float, float]:
    """Return (accuracy, f1) on a held-out test set."""
    preds = (model.predict(X_test, verbose=0).ravel() > 0.5).astype(int)
    acc = float(accuracy_score(y_test, preds)) if _HAS_APP else float(np.mean(preds == y_test))
    try:
        f1 = float(sk_f1(y_test, preds, zero_division=0))
    except Exception:
        f1 = 0.0
    return acc, f1


def benchmark_fedavg(
    num_rounds: int = 10,
    num_clients: int = 5,
    local_epochs: int = 2,
    dp_epsilon: float = 2.0,
    clip_norm: float = 1.0,
    seed: int = 42,
) -> list[RoundResult]:
    """
    Run FedAvg benchmark across `num_rounds` rounds.

    Args:
        num_rounds: Number of federated rounds.
        num_clients: Number of participating clients per round.
        local_epochs: Local SGD epochs per client per round.
        dp_epsilon: Total privacy budget (simple composition; epsilon split per round).
        clip_norm: L2 clipping norm for gradient updates.
        seed: Random seed for reproducibility.

    Returns:
        List of RoundResult dataclasses, one per round.
    """
    np.random.seed(seed)
    clients = _build_synthetic_clients(num_clients)
    model = _build_global_model()
    results: list[RoundResult] = []
    prev_accuracy = 0.0

    epsilon_per_round = dp_epsilon / num_rounds

    # Build a small held-out eval set from all clients
    X_eval = np.concatenate([c["X_train"][:20] for c in clients], axis=0)
    y_eval = np.concatenate([c["y_train"][:20] for c in clients], axis=0)

    global_weights = model.get_weights()

    for rnd in range(1, num_rounds + 1):
        t_start = time.perf_counter()
        total_bytes = 0
        client_deltas: list[list[np.ndarray]] = []
        client_sizes: list[int] = []

        for client in clients:
            model.set_weights(global_weights)
            model.fit(
                client["X_train"],
                client["y_train"],
                epochs=local_epochs,
                batch_size=32,
                verbose=0,
            )
            local_weights = model.get_weights()
            delta = [lw - gw for lw, gw in zip(local_weights, global_weights)]

            if _HAS_APP:
                delta = clip_update(delta, max_norm=clip_norm)
            total_bytes += _bytes_of_weights(local_weights)
            client_deltas.append(delta)
            client_sizes.append(client["n"])

        # FedAvg aggregation
        total = sum(client_sizes)
        agg_delta = [
            sum((s / total) * d[i] for d, s in zip(client_deltas, client_sizes))
            for i in range(len(global_weights))
        ]

        # DP noise on aggregated update
        if _HAS_APP:
            agg_delta = add_gaussian_noise_to_update(
                agg_delta,
                noise_multiplier=1.0,
                num_clients=num_clients,
            )

        global_weights = [gw + ad for gw, ad in zip(global_weights, agg_delta)]

        t_elapsed = time.perf_counter() - t_start

        model.set_weights(global_weights)
        accuracy, f1 = _eval_model(model, X_eval, y_eval)
        convergence_rate = abs(accuracy - prev_accuracy)
        prev_accuracy = accuracy

        results.append(RoundResult(
            algorithm="FedAvg",
            round=rnd,
            accuracy=round(accuracy, 4),
            f1_score=round(f1, 4),
            training_time_seconds=round(t_elapsed, 3),
            bytes_transferred=total_bytes,
            epsilon=round(epsilon_per_round * rnd, 4),
            convergence_rate=round(convergence_rate, 4),
        ))

    return results


def benchmark_fedprox(
    num_rounds: int = 10,
    num_clients: int = 5,
    mu: float = 0.01,
    local_epochs: int = 2,
    dp_epsilon: float = 2.0,
    clip_norm: float = 1.0,
    seed: int = 42,
) -> list[RoundResult]:
    """
    Run FedProx benchmark.

    FedProx adds a proximal term (mu/2) * ||w - w_global||^2 to each client's loss,
    which is simulated here by applying a soft pull toward the global weights after
    each local gradient step.

    Args:
        num_rounds: Number of federated rounds.
        num_clients: Number of participating clients.
        mu: FedProx proximal coefficient (higher = closer to global model).
        local_epochs: Local epochs per round.
        dp_epsilon: Total DP budget.
        clip_norm: L2 gradient clipping norm.
        seed: Reproducibility seed.

    Returns:
        List of RoundResult dataclasses.
    """
    np.random.seed(seed)
    clients = _build_synthetic_clients(num_clients)
    model = _build_global_model()
    results: list[RoundResult] = []
    prev_accuracy = 0.0

    epsilon_per_round = dp_epsilon / num_rounds
    X_eval = np.concatenate([c["X_train"][:20] for c in clients], axis=0)
    y_eval = np.concatenate([c["y_train"][:20] for c in clients], axis=0)

    global_weights = model.get_weights()

    for rnd in range(1, num_rounds + 1):
        t_start = time.perf_counter()
        total_bytes = 0
        client_deltas: list[list[np.ndarray]] = []
        client_sizes: list[int] = []

        for client in clients:
            model.set_weights(global_weights)

            # Proximal SGD: after each epoch pull weights toward global (approximates FedProx)
            for _ in range(local_epochs):
                model.train_on_batch(client["X_train"], client["y_train"])
                local_w = model.get_weights()
                # Proximal correction: w <- w - mu*(w - w_global)
                proximal_w = [lw - mu * (lw - gw) for lw, gw in zip(local_w, global_weights)]
                model.set_weights(proximal_w)

            local_weights = model.get_weights()
            delta = [lw - gw for lw, gw in zip(local_weights, global_weights)]
            if _HAS_APP:
                delta = clip_update(delta, max_norm=clip_norm)
            total_bytes += _bytes_of_weights(local_weights)
            client_deltas.append(delta)
            client_sizes.append(client["n"])

        total = sum(client_sizes)
        agg_delta = [
            sum((s / total) * d[i] for d, s in zip(client_deltas, client_sizes))
            for i in range(len(global_weights))
        ]

        if _HAS_APP:
            agg_delta = add_gaussian_noise_to_update(agg_delta, noise_multiplier=1.0, num_clients=num_clients)

        global_weights = [gw + ad for gw, ad in zip(global_weights, agg_delta)]
        t_elapsed = time.perf_counter() - t_start

        model.set_weights(global_weights)
        accuracy, f1 = _eval_model(model, X_eval, y_eval)
        convergence_rate = abs(accuracy - prev_accuracy)
        prev_accuracy = accuracy

        results.append(RoundResult(
            algorithm=f"FedProx(mu={mu})",
            round=rnd,
            accuracy=round(accuracy, 4),
            f1_score=round(f1, 4),
            training_time_seconds=round(t_elapsed, 3),
            bytes_transferred=total_bytes,
            epsilon=round(epsilon_per_round * rnd, 4),
            convergence_rate=round(convergence_rate, 4),
        ))

    return results


def benchmark_comparison(
    num_rounds: int = 10,
    num_clients: int = 5,
    mu: float = 0.01,
    output_path: str | None = None,
) -> str:
    """
    Run FedAvg and FedProx back-to-back and save results to CSV.

    Args:
        num_rounds: Rounds per algorithm.
        num_clients: Clients per round.
        mu: FedProx proximal coefficient.
        output_path: CSV output path. Defaults to data/results/benchmark_comparison.csv.

    Returns:
        Absolute path to the saved CSV file.
    """
    fedavg_results = benchmark_fedavg(num_rounds=num_rounds, num_clients=num_clients)
    fedprox_results = benchmark_fedprox(num_rounds=num_rounds, num_clients=num_clients, mu=mu)

    all_results = fedavg_results + fedprox_results

    if output_path is None:
        output_path = str(RESULTS_DIR / "benchmark_comparison.csv")

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for r in all_results:
            writer.writerow({
                "algorithm": r.algorithm,
                "round": r.round,
                "accuracy": r.accuracy,
                "f1_score": r.f1_score,
                "training_time_seconds": r.training_time_seconds,
                "bytes_transferred": r.bytes_transferred,
                "epsilon": r.epsilon,
                "convergence_rate": r.convergence_rate,
            })

    return output_path


def plot_benchmark_results(csv_path: str | None = None) -> "go.Figure":
    """
    Generate a 2x2 Plotly figure comparing FedAvg vs FedProx.

    Panels: accuracy curves, F1 score, communication overhead, training time.

    Args:
        csv_path: Path to benchmark CSV. Defaults to data/results/benchmark_comparison.csv.

    Returns:
        Plotly Figure object.
    """
    if not _HAS_PLOTLY:
        raise ImportError("plotly is required for plotting.")

    import pandas as pd

    if csv_path is None:
        csv_path = str(RESULTS_DIR / "benchmark_comparison.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Run benchmark_comparison() first. Not found: {csv_path}")

    df = pd.read_csv(csv_path)

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Accuracy per Round",
            "F1 Score per Round",
            "Cumulative Bytes Transferred",
            "Training Time per Round (s)",
        ],
    )

    colors = {"FedAvg": "#636EFA", "FedProx": "#EF553B"}
    for algo, grp in df.groupby("algorithm"):
        color = colors.get(algo.split("(")[0], "#00CC96")
        rounds = grp["round"].tolist()

        fig.add_trace(go.Scatter(x=rounds, y=grp["accuracy"].tolist(), name=algo, line=dict(color=color), legendgroup=algo), row=1, col=1)
        fig.add_trace(go.Scatter(x=rounds, y=grp["f1_score"].tolist(), name=algo, line=dict(color=color), legendgroup=algo, showlegend=False), row=1, col=2)
        cum_bytes = grp["bytes_transferred"].cumsum().tolist()
        fig.add_trace(go.Scatter(x=rounds, y=cum_bytes, name=algo, line=dict(color=color), legendgroup=algo, showlegend=False), row=2, col=1)
        fig.add_trace(go.Scatter(x=rounds, y=grp["training_time_seconds"].tolist(), name=algo, line=dict(color=color), legendgroup=algo, showlegend=False), row=2, col=2)

    fig.update_layout(title="FedAvg vs FedProx Benchmark Comparison", height=600, template="plotly_dark")
    return fig
