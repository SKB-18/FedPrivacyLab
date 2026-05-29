"""Compare privacy mechanisms against utility metrics (spec § Privacy and Utility Evaluator)."""

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class UtilityReport:
    experiment_id: int
    mode: str
    mean_analytics_relative_error: float | None
    final_accuracy: float | None
    final_roc_auc: float | None
    privacy_score: float
    utility_score: float
    federated_vs_true_gap: float | None
    notes: list[str]


def privacy_score_from_mode(mode: str, dp_enabled: bool, secure_agg: bool) -> float:
    if mode == "centralized_baseline":
        return 0.1
    if mode == "fedavg_secureagg_dp" or (dp_enabled and secure_agg):
        return 0.85
    if mode == "fedavg_secureagg" or secure_agg:
        return 0.65
    if mode == "federated_analytics":
        return 0.7 if dp_enabled else 0.55
    if mode == "fedavg":
        return 0.35
    return 0.5


def evaluate_utility(
    experiment_id: int,
    mode: str,
    analytics_results: list[dict],
    round_metrics: list[dict],
    dp_enabled: bool = False,
    secure_agg: bool = False,
) -> UtilityReport:
    rel_errors = [
        r["relative_error"]
        for r in analytics_results
        if r.get("relative_error") is not None and not r.get("suppressed")
    ]
    mean_rel = float(np.mean(rel_errors)) if rel_errors else None

    final_acc = None
    final_auc = None
    if round_metrics:
        last = round_metrics[-1]
        final_acc = last.get("accuracy")
        final_auc = last.get("roc_auc")

    p_score = privacy_score_from_mode(mode, dp_enabled, secure_agg)
    # Utility: higher accuracy / lower error = better
    u_components = []
    if final_acc is not None:
        u_components.append(final_acc)
    if mean_rel is not None:
        u_components.append(max(0.0, 1.0 - mean_rel))
    utility_score = float(np.mean(u_components)) if u_components else 0.5

    fed_gaps = [
        abs(r.get("federated_value", 0) - r.get("true_value", 0))
        for r in analytics_results
        if r.get("true_value") is not None and not r.get("suppressed")
    ]
    gap = float(np.mean(fed_gaps)) if fed_gaps else None

    notes = []
    if gap is not None and gap < 0.05:
        notes.append("Federated aggregates closely match centralized truth (no DP).")
    if dp_enabled and mean_rel and mean_rel > 0.1:
        notes.append("DP noise increases analytics error as expected for privacy.")
    if mode.startswith("fedavg") and final_acc and final_acc > 0.7:
        notes.append("Model utility acceptable for binary poor-experience prediction.")

    return UtilityReport(
        experiment_id=experiment_id,
        mode=mode,
        mean_analytics_relative_error=mean_rel,
        final_accuracy=final_acc,
        final_roc_auc=final_auc,
        privacy_score=p_score,
        utility_score=utility_score,
        federated_vs_true_gap=gap,
        notes=notes,
    )
