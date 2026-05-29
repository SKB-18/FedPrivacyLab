"""
Generate canonical graphs -> data/results/graphs/latest/

Learning: separate charts for real HDFS and synthetic scale.
Analytics: live pipeline (true = federated without DP; DP error explicit).
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sqlalchemy import create_engine

from app.analytics.histogram import (
    LATENCY_BUCKET_ORDER,
    histogram_percentages,
    latency_bucket,
)
from app.analytics.metric_types import (
    LATENCY_METRICS,
    RATE_METRICS,
    metric_display_name,
)

DP_ERROR_METRICS = RATE_METRICS | LATENCY_METRICS

OUT_BASE = ROOT / "data" / "results" / "graphs"
LATEST_DIR = OUT_BASE / "latest"
DB_PATH = ROOT / "data" / "fedprivacylab.db"

MODE_LABELS = {
    "centralized_baseline": "Centralized baseline",
    "fedavg": "FedAvg",
    "fedavg_secureagg": "FedAvg + SecAgg",
    "fedavg_secureagg_dp": "FedAvg + SecAgg + DP",
}

LEARNING_MODES = {"fedavg", "fedavg_secureagg", "fedavg_secureagg_dp", "centralized_baseline"}

SHORT_MODE_LABELS = {
    "Centralized baseline": "Centralized",
    "FedAvg": "FedAvg",
    "FedAvg + SecAgg": "+ SecAgg",
    "FedAvg + SecAgg + DP": "+ SecAgg + DP",
}


def _axis_range(
    values: pd.Series | list[float],
    *,
    floor: float = 0.0,
    cap: float | None = 1.0,
    pad_frac: float = 0.12,
    min_span: float = 0.05,
) -> list[float]:
    """Y-axis limits that zoom in when values cluster (rates, F1, accuracy)."""
    vals = [float(v) for v in values if v is not None and pd.notna(v)]
    if not vals:
        return [floor, 1.0 if cap is None else min(1.0, cap)]
    lo, hi = min(vals), max(vals)
    span = max(hi - lo, min_span)
    pad = max(span * pad_frac, hi * 0.03, 0.01)
    ymin = max(floor, lo - pad * 0.35)
    ymax = hi + pad
    if cap is not None:
        ymax = min(cap, ymax)
    if ymax <= ymin:
        ymax = ymin + min_span
    return [ymin, ymax]


def _style(fig: go.Figure, height: int = 540, bottom_margin: int = 110) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=64, r=24, t=96, b=bottom_margin),
        font=dict(size=13),
        title_x=0.5,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    )
    return fig


def _save(fig: go.Figure, out_dir: Path, name: str) -> list[str]:
    if not fig.layout.margin or fig.layout.margin.t is None:
        _style(fig)
    paths = []
    html = out_dir / f"{name}.html"
    fig.write_html(str(html), include_plotlyjs="cdn")
    paths.append(html.name)
    try:
        png = out_dir / f"{name}.png"
        fig.write_image(str(png), width=1200, height=680, scale=2)
        paths.append(png.name)
    except Exception as exc:
        print(f"Warning: PNG export failed for {name}: {exc}")
    return paths


def load_db() -> dict[str, pd.DataFrame]:
    if not DB_PATH.exists():
        return {}
    engine = create_engine(f"sqlite:///{DB_PATH}")
    out = {}
    for t in ["experiments", "round_metrics", "analytics_results"]:
        try:
            out[t] = pd.read_sql_table(t, engine)
        except Exception:
            out[t] = pd.DataFrame()
    return out


def _batch_learning_stats(
    batch: pd.DataFrame, metrics: pd.DataFrame, need: set[str]
) -> dict[str, float] | None:
    """Final-round accuracy/ROC-AUC per learning mode for batch quality scoring."""
    stats: dict[str, float] = {}
    for mode in need:
        rows = batch[batch["mode"] == mode]
        if rows.empty:
            return None
        eid = int(rows.iloc[0]["id"])
        g = metrics[metrics["experiment_id"] == eid].sort_values("round_number")
        if g.empty:
            return None
        last = g.iloc[-1]
        stats[f"{mode}_acc"] = float(last.get("accuracy") or 0)
        auc = last.get("roc_auc")
        if auc is None or (isinstance(auc, float) and pd.isna(auc)):
            return None
        stats[f"{mode}_auc"] = float(auc)
    return stats


def _batch_coherent(batch: pd.DataFrame, metrics: pd.DataFrame) -> bool:
    """All modes must share the same configured rounds and complete metric rows."""
    if batch.empty or "rounds" not in batch.columns:
        return False
    rounds = batch["rounds"].dropna().unique()
    if len(rounds) != 1:
        return False
    target = int(rounds[0])
    if target < 2:
        return False
    for eid in batch["id"].astype(int):
        g = metrics[metrics["experiment_id"] == eid]
        if g.empty or int(g["round_number"].max()) < target:
            return False
    return True


def _batch_quality_ok(stats: dict[str, float]) -> bool:
    fed, sec, dp = stats["fedavg_acc"], stats["fedavg_secureagg_acc"], stats["fedavg_secureagg_dp_acc"]
    if min(fed, sec, dp) < 0.5:
        return False
    if abs(fed - sec) > 0.15:
        return False
    if abs(stats["fedavg_auc"] - stats["fedavg_secureagg_auc"]) > 0.22:
        return False
    if dp < fed - 0.3:
        return False
    return True


def latest_comparison_experiments(
    exp: pd.DataFrame, dataset: str, metrics: pd.DataFrame | None = None
) -> pd.DataFrame:
    """
    Pick one complete comparison batch (all modes from the same run).

    Using groupby(mode).tail(1) mixes experiments from different runs and
    produces misleading training curves (e.g. SecAgg from an older failed run).
    """
    if exp.empty:
        return exp
    sub = exp[
        exp["name"].astype(str).str.startswith("comparison_")
        & (exp["dataset"].astype(str) == dataset)
        & (exp["status"].astype(str) == "completed")
    ].copy()
    if sub.empty:
        return sub

    anchors = sub[sub["mode"] == "centralized_baseline"].sort_values("id")
    if anchors.empty:
        anchors = sub[sub["mode"] == "fedavg"].sort_values("id")
    need = {"fedavg", "fedavg_secureagg", "fedavg_secureagg_dp"}
    picked = None
    best_key: tuple[float, float] | None = None
    best_effort: pd.DataFrame | None = None
    best_effort_key: tuple[float, float] | None = None
    if not anchors.empty and metrics is not None and not metrics.empty:
        for anchor_id in anchors["id"].iloc[::-1]:
            window = sub[(sub["id"] >= anchor_id) & (sub["id"] <= anchor_id + 24)]
            batch = window.sort_values("id").groupby("mode", as_index=False).last()
            if not need.issubset(set(batch["mode"])):
                continue
            if not _batch_coherent(batch, metrics):
                continue
            stats = _batch_learning_stats(batch, metrics, need)
            if stats is None:
                continue
            if not _batch_quality_ok(stats):
                effort_key = (
                    abs(stats["fedavg_acc"] - stats["fedavg_secureagg_acc"]),
                    -min(
                        stats["fedavg_acc"],
                        stats["fedavg_secureagg_acc"],
                        stats["fedavg_secureagg_dp_acc"],
                    ),
                )
                if best_effort_key is None or effort_key < best_effort_key:
                    best_effort_key = effort_key
                    best_effort = batch
                continue
            key = (
                abs(stats["fedavg_acc"] - stats["fedavg_secureagg_acc"])
                + abs(stats["fedavg_auc"] - stats["fedavg_secureagg_auc"]) * 0.5,
                -min(stats["fedavg_acc"], stats["fedavg_secureagg_acc"], stats["fedavg_secureagg_dp_acc"]),
            )
            if best_key is None or key < best_key:
                best_key = key
                picked = batch
    if picked is not None:
        sub = picked
    elif best_effort is not None:
        sub = best_effort
    else:
        sub = sub.sort_values("id").groupby("mode").tail(1)

    sub["mode_label"] = sub["mode"].map(lambda m: MODE_LABELS.get(m, m))
    return sub


def _final_round_rows(metrics: pd.DataFrame, exp_ids: list[int]) -> pd.DataFrame:
    """Last round metrics — matches comparison_summary.csv (spec-aligned reporting)."""
    rows = []
    for eid in exp_ids:
        g = metrics[metrics["experiment_id"] == eid].sort_values("round_number")
        if not g.empty:
            rows.append(g.iloc[-1])
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def _rel_err(true_v: float, est_v: float) -> float:
    if true_v == 0:
        return 0.0 if est_v == 0 else 1.0
    return abs(est_v - true_v) / abs(true_v)


def _bucket_counts(latencies: list[float]) -> dict[str, int]:
    hist = {b: 0 for b in LATENCY_BUCKET_ORDER}
    for lat in latencies:
        hist[latency_bucket(float(lat))] += 1
    return hist


def _real_hdfs_eval_caption(dataset: str = "real_hdfs_loghub") -> str:
    """Describe holdout size so 100% bars are not misread."""
    from app.config_loader import set_active_dataset
    from app.coordinator.client_registry import (
        clear_registry,
        configure_train_eval_split,
        get_eval_meta,
        get_eval_records,
        populate_registry,
    )
    from app.simulation.real_telemetry_loader import load_real_hdfs_clients

    set_active_dataset(dataset)
    n_clients = len(load_real_hdfs_clients(partition_by="pid" if "pid" in dataset else "component"))
    eid = 9_001
    populate_registry(eid, num_clients=n_clients, dataset=dataset)
    configure_train_eval_split(eid, seed=42, dataset=dataset)
    meta = get_eval_meta(eid)
    n = len(get_eval_records(eid))
    clear_registry(eid)
    if not meta:
        return ""
    return (
        f"holdout: {meta.get('num_eval_records', n)} records "
        f"({meta.get('split', '?')} split), "
        f"{meta.get('eval_positive_rate', 0):.1%} positive"
    )


def _privacy_strength(mode: str, secure_agg: bool, dp_enabled: bool, epsilon: float | None) -> float:
    """Ordinal privacy strength for scatter (higher = more private)."""
    if mode == "centralized_baseline":
        return 0.0
    strength = 0.25 if mode == "fedavg" else 0.0
    if secure_agg:
        strength += 0.5
    if dp_enabled or "dp" in mode:
        strength += 1.0 / max(float(epsilon or 2.0), 0.1)
    return strength


def _rate_metric_rows(live_no_dp: list[dict], live_dp: list[dict]) -> list[dict]:
    """Global rate metrics with true (no DP), federated, and DP values."""
    dp_map = {r["metric_name"]: r for r in live_dp if not r.get("feature")}
    rows = []
    for r in live_no_dp:
        if r.get("feature") or r.get("suppressed") or r["metric_name"] not in RATE_METRICS:
            continue
        d = dp_map.get(r["metric_name"], {})
        tv, fv = float(r["true_value"]), float(r["federated_value"])
        dv = d.get("dp_noisy_value")
        rows.append({
            "metric": metric_display_name(r["metric_name"]),
            "true": tv,
            "federated": fv,
            "dp": float(dv) if dv is not None and pd.notna(dv) else None,
            "fed_matches": abs(tv - fv) < 1e-9,
        })
    return rows


def live_real_analytics(dp_enabled: bool, epsilon: float = 2.0) -> tuple[list[dict], dict, int]:
    from app.config_loader import set_active_dataset
    from app.analytics.federated_analytics import run_analytics_round
    from app.simulation.real_telemetry_loader import load_real_hdfs_clients

    set_active_dataset("real_hdfs_loghub")
    clients = load_real_hdfs_clients(partition_by="component")
    records_map = {c.client_id: [r.to_dict() for r in c.records] for c in clients}
    cohort_map = {c.client_id: f"{c.locale}_{c.device_tier}" for c in clients}
    ids = list(records_map.keys())
    results, meta = run_analytics_round(
        records_map, ids, dp_enabled=dp_enabled, epsilon=epsilon, cohort_map=cohort_map
    )
    rows = [
        {
            "metric_name": r.metric_name,
            "feature": r.feature,
            "true_value": r.true_value,
            "federated_value": r.federated_value,
            "dp_noisy_value": r.dp_noisy_value,
            "relative_error": r.relative_error,
            "suppressed": r.suppressed,
        }
        for r in results
        if not r.feature
    ]
    return rows, meta, len(ids)


def chart_learning_modes(
    exp: pd.DataFrame, metrics: pd.DataFrame, out: Path, dataset: str, stem: str
) -> list[str]:
    cmp_exp = latest_comparison_experiments(exp, dataset, metrics)
    cmp_exp = cmp_exp[cmp_exp["mode"].isin(LEARNING_MODES)]
    if cmp_exp.empty or metrics.empty:
        return []
    ids = cmp_exp["id"].tolist()
    last = _final_round_rows(metrics, ids)
    if last.empty:
        return []
    merged = last.merge(
        cmp_exp[["id", "mode", "mode_label", "num_clients"]],
        left_on="experiment_id",
        right_on="id",
    )
    merged["short_label"] = merged["mode_label"].map(lambda m: SHORT_MODE_LABELS.get(m, m))
    mode_order = [
        "centralized_baseline",
        "fedavg",
        "fedavg_secureagg",
        "fedavg_secureagg_dp",
    ]
    merged["_ord"] = merged["mode"].map({m: i for i, m in enumerate(mode_order)})
    merged = merged.sort_values("_ord")
    if "real" in dataset:
        part = "13 pid clients" if "pid" in dataset else "5 subsystems"
        label = f"real HDFS ({part}) — {_real_hdfs_eval_caption(dataset)}"
    else:
        label = f"synthetic ({int(merged['num_clients'].iloc[0])} clients)"
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=("Accuracy", "ROC-AUC", "F1"),
        horizontal_spacing=0.08,
    )
    colors = px.colors.qualitative.Set2
    for i, col in enumerate(["accuracy", "roc_auc", "f1_score"], start=1):
        vals = merged[col].fillna(0)
        fig.add_trace(
            go.Bar(
                x=merged["short_label"],
                y=vals,
                marker_color=colors[i - 1],
                text=[
                    f"{v:.2%}" if col != "roc_auc" else f"{v:.3f}"
                    for v in vals
                ],
                textposition="outside",
                cliponaxis=False,
                showlegend=False,
            ),
            row=1,
            col=i,
        )
        if col == "roc_auc" or "real" in dataset:
            fig.update_yaxes(range=[0, 1.05], row=1, col=i)
        else:
            fig.update_yaxes(range=_axis_range(vals, cap=1.05), row=1, col=i)
    _style(fig, height=560, bottom_margin=90)
    fig.update_layout(
        title=f"Learning modes ({label}) — final round, holdout eval (F1 uses tuned threshold)",
    )
    return _save(fig, out, stem)


def chart_training_curves(
    exp: pd.DataFrame, metrics: pd.DataFrame, out: Path, dataset: str, stem: str
) -> list[str]:
    cmp_exp = latest_comparison_experiments(exp, dataset, metrics)
    cmp_exp = cmp_exp[cmp_exp["mode"].isin(LEARNING_MODES)]
    if cmp_exp.empty:
        return []
    m = metrics[metrics["experiment_id"].isin(cmp_exp["id"])].merge(
        cmp_exp[["id", "mode_label"]], left_on="experiment_id", right_on="id"
    )
    if m.empty:
        return []
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        subplot_titles=("Accuracy by round", "ROC-AUC by round (threshold-free)"),
        vertical_spacing=0.12,
    )
    colors = px.colors.qualitative.Set2
    rounds = sorted(m["round_number"].unique())
    mode_order = []
    present = set(m["mode_label"].unique())
    for mode_key in (
        "centralized_baseline",
        "fedavg",
        "fedavg_secureagg",
        "fedavg_secureagg_dp",
    ):
        label = MODE_LABELS.get(mode_key, mode_key)
        if label in present:
            mode_order.append(label)
    for i, mode in enumerate(mode_order):
        sub = m[m["mode_label"] == mode].sort_values("round_number")
        c = colors[i % len(colors)]
        if len(sub) == 1 and "Centralized" in mode:
            fig.add_trace(
                go.Scatter(
                    x=rounds, y=[sub["accuracy"].iloc[0]] * len(rounds),
                    mode="lines", name=mode, line=dict(color=c, dash="dash"),
                    legendgroup=mode,
                ),
                row=1, col=1,
            )
            auc0 = sub["roc_auc"].iloc[0]
            fig.add_trace(
                go.Scatter(
                    x=rounds,
                    y=[auc0] * len(rounds) if pd.notna(auc0) else [None] * len(rounds),
                    mode="lines", showlegend=False, line=dict(color=c, dash="dash"),
                    legendgroup=mode,
                ),
                row=2, col=1,
            )
            continue
        fig.add_trace(
            go.Scatter(x=sub["round_number"], y=sub["accuracy"], mode="lines+markers",
                       name=mode, line=dict(color=c), legendgroup=mode),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=sub["round_number"],
                y=sub["roc_auc"],
                mode="lines+markers",
                connectgaps=False,
                name=mode,
                line=dict(color=c),
                showlegend=False,
                legendgroup=mode,
            ),
            row=2, col=1,
        )
    y_full = [0, 1.05] if "real" in dataset else None
    if y_full:
        fig.update_yaxes(range=y_full, row=1, col=1)
        fig.update_yaxes(range=y_full, row=2, col=1)
    else:
        acc_vals = m["accuracy"].dropna()
        auc_vals = m["roc_auc"].dropna()
        fig.update_yaxes(range=_axis_range(acc_vals, cap=1.05), row=1, col=1)
        fig.update_yaxes(
            range=_axis_range(auc_vals, cap=1.05, min_span=0.08), row=2, col=1
        )
    fig.update_xaxes(dtick=1, row=2, col=1)
    if "real" in dataset:
        ds = f"real HDFS — {_real_hdfs_eval_caption(dataset)}"
    else:
        ds = "synthetic 200-client"
    _style(fig, height=640, bottom_margin=80)
    fig.update_layout(
        title=(
            f"Training curves ({ds}) — accuracy + ROC-AUC on holdout per round"
        ),
    )
    return _save(fig, out, stem)


def chart_fedavg_improvement(
    exp: pd.DataFrame, metrics: pd.DataFrame, out: Path, dataset: str, stem: str
) -> list[str]:
    cmp_exp = latest_comparison_experiments(exp, dataset, metrics)
    fed = cmp_exp[cmp_exp["mode"] == "fedavg"]
    if fed.empty:
        return []
    rows = metrics[metrics["experiment_id"] == int(fed.iloc[0]["id"])].sort_values("round_number")
    if len(rows) < 2:
        return []
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Accuracy", "F1"))
    fig.add_trace(
        go.Scatter(x=rows["round_number"], y=rows["accuracy"], mode="lines+markers+text",
                   text=[f"{a:.1%}" for a in rows["accuracy"]], textposition="top center",
                   line=dict(color="#2980b9", width=3), name="Accuracy"),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=rows["round_number"], y=rows["f1_score"].fillna(0), mode="lines+markers+text",
                   text=[f"{f:.3f}" for f in rows["f1_score"].fillna(0)], textposition="top center",
                   line=dict(color="#8e44ad", width=3), name="F1"),
        row=1, col=2,
    )
    fig.update_yaxes(range=_axis_range(rows["accuracy"], cap=1.05), row=1, col=1)
    fig.update_yaxes(range=_axis_range(rows["f1_score"].fillna(0), cap=1.05), row=1, col=2)
    n = int(fed.iloc[0]["num_clients"])
    if "real" in dataset:
        ds = f"real HDFS — {_real_hdfs_eval_caption(dataset)}"
    else:
        ds = f"synthetic ({n} clients)"
    fig.update_layout(title=f"FedAvg round-over-round ({ds})", template="plotly_white", height=460)
    return _save(fig, out, stem)


def chart_privacy_utility(
    exp: pd.DataFrame, metrics: pd.DataFrame, out: Path, dataset: str, stem: str
) -> list[str]:
    cmp_exp = latest_comparison_experiments(exp, dataset, metrics)
    learning = cmp_exp[cmp_exp["mode"].isin(LEARNING_MODES)]
    if learning.empty:
        return []
    last = _final_round_rows(metrics, learning["id"].tolist())
    if last.empty:
        return []
    merged = last.merge(learning, left_on="experiment_id", right_on="id")
    merged["short_label"] = merged["mode_label"].map(lambda m: SHORT_MODE_LABELS.get(m, m))
    merged["privacy_strength"] = merged.apply(
        lambda r: _privacy_strength(
            str(r["mode"]),
            bool(r.get("secure_agg_enabled")),
            bool(r.get("dp_enabled")) or "dp" in str(r["mode"]),
            r.get("epsilon"),
        ),
        axis=1,
    )
    ds = "real HDFS (5 clients)" if "real" in dataset else "synthetic 200-client"
    # Avoid overlapping scatter when all accuracies coincide (common on tiny real HDFS).
    eval_note = f" — {_real_hdfs_eval_caption(dataset)}" if "real" in dataset else ""
    if merged["accuracy"].nunique() <= 1 and len(merged) > 1 and "real" not in dataset:
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Accuracy", "F1"))
        fig.add_trace(
            go.Bar(x=merged["short_label"], y=merged["accuracy"], marker_color="#3498db",
                   text=[f"{v:.1%}" for v in merged["accuracy"]], textposition="outside", cliponaxis=False),
            row=1, col=1,
        )
        fig.add_trace(
            go.Bar(x=merged["short_label"], y=merged["f1_score"].fillna(0), marker_color="#9b59b6",
                   text=[f"{v:.1%}" for v in merged["f1_score"].fillna(0)], textposition="outside", cliponaxis=False),
            row=1, col=2,
        )
        fig.update_yaxes(range=_axis_range(merged["accuracy"], cap=1.08), row=1, col=1)
        fig.update_yaxes(
            range=_axis_range(merged["f1_score"].fillna(0), cap=1.08), row=1, col=2
        )
        _style(fig, height=500, bottom_margin=90)
        fig.update_layout(title=f"Privacy modes vs metrics ({ds}){eval_note}")
    elif "real" in dataset:
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Accuracy (final round)", "F1 (final round)"))
        fig.add_trace(
            go.Bar(x=merged["short_label"], y=merged["accuracy"], marker_color="#3498db",
                   text=[f"{v:.1%}" for v in merged["accuracy"]], textposition="outside", cliponaxis=False),
            row=1, col=1,
        )
        fig.add_trace(
            go.Bar(x=merged["short_label"], y=merged["f1_score"].fillna(0), marker_color="#9b59b6",
                   text=[f"{v:.1%}" for v in merged["f1_score"].fillna(0)], textposition="outside", cliponaxis=False),
            row=1, col=2,
        )
        fig.update_yaxes(range=_axis_range(merged["accuracy"], cap=1.08), row=1, col=1)
        fig.update_yaxes(
            range=_axis_range(merged["f1_score"].fillna(0), cap=1.08), row=1, col=2
        )
        _style(fig, height=500, bottom_margin=90)
        fig.update_layout(
            title=f"Real HDFS learning ({eval_note}) — use synthetic charts for scale/privacy tradeoffs",
        )
    else:
        fig = px.scatter(
            merged, x="privacy_strength", y="accuracy", color="mode_label",
            text="short_label", size=merged["f1_score"].fillna(0.1).clip(lower=0.05),
            title=f"Privacy strength vs accuracy ({ds}) — final round, higher x = more private",
            labels={"privacy_strength": "Privacy strength (ordinal)", "accuracy": "Accuracy"},
        )
        _style(fig, height=540, bottom_margin=70)
        fig.update_traces(textposition="top center")
        fig.update_xaxes(dtick=0.25)
    return _save(fig, out, stem)


def chart_analytics_rates(live_no_dp: list[dict], live_dp: list[dict], out: Path) -> list[str]:
    rows = _rate_metric_rows(live_no_dp, live_dp)
    if not rows:
        return []
    n = len(rows)
    fig = make_subplots(
        rows=1, cols=n,
        subplot_titles=[r["metric"] for r in rows],
        horizontal_spacing=0.08,
    )
    for i, row in enumerate(rows, start=1):
        tv, fv = row["true"], row["federated"]
        dv = row["dp"] if row["dp"] is not None else tv
        vals = [tv, fv, dv]
        lo, hi = min(vals), max(vals)
        span = hi - lo
        ymax = hi + max(span * 2.5, hi * 0.06, 0.004)
        fig.add_trace(
            go.Bar(
                x=["True", "Federated", "DP noisy"],
                y=vals,
                marker_color=["#27ae60", "#2980b9", "#c0392b"],
                text=[f"{v:.2%}" for v in vals],
                textposition="outside",
                showlegend=False,
            ),
            row=1, col=i,
        )
        fig.update_yaxes(tickformat=".2%", range=[max(0, lo - span * 0.5), ymax], row=1, col=i)
    _style(fig, height=540, bottom_margin=70)
    fig.update_layout(
        title="Analytics rates: federated matches true; DP (ε=2) adds noise",
        annotations=[
            dict(
                text="Without DP, federated estimates equal centralized truth",
                xref="paper", yref="paper", x=0.5, y=1.06,
                showarrow=False, font=dict(size=11, color="#2980b9"),
            )
        ],
    )
    return _save(fig, out, "C_analytics_rates")


def chart_analytics_true_fed_dp(live_no_dp: list[dict], live_dp: list[dict], out: Path) -> list[str]:
    rate_rows = _rate_metric_rows(live_no_dp, live_dp)
    dp_map = {r["metric_name"]: r for r in live_dp if not r.get("feature")}
    rate_labels, dp_errs = [], []
    for row in rate_rows:
        rate_labels.append(row["metric"])
        dp_errs.append(_rel_err(row["true"], row["dp"]) * 100 if row["dp"] is not None else 0)
    lat_row = None
    for r in live_no_dp:
        if r.get("feature") or r.get("suppressed"):
            continue
        if r["metric_name"] in LATENCY_METRICS:
            d = dp_map.get(r["metric_name"], {})
            tv = float(r["true_value"])
            dv = d.get("dp_noisy_value")
            lat_row = (tv, float(dv) if dv is not None and pd.notna(dv) else tv)
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Rate metrics: DP error vs true (%)", "Avg latency: true vs DP (ms)"),
    )
    if rate_labels:
        fig.add_trace(
            go.Bar(x=rate_labels, y=dp_errs, marker_color="#c0392b", name="DP error %",
                   text=[f"{e:.1f}%" for e in dp_errs], textposition="outside"),
            row=1, col=1,
        )
        fig.update_yaxes(title_text="Relative error (%)", row=1, col=1)
    if lat_row:
        tv, dv = lat_row
        fig.add_trace(
            go.Bar(
                x=["True", "DP noisy"], y=[tv, dv],
                marker_color=["#27ae60", "#c0392b"],
                text=[f"{tv:.0f}", f"{dv:.0f}"], textposition="outside", showlegend=False,
            ),
            row=1, col=2,
        )
        if abs(tv - dv) < 1:
            fig.add_annotation(text="Federated ≡ True", x=0.5, y=1.1, showarrow=False, row=1, col=2)
        fig.update_yaxes(title_text="ms", row=1, col=2)
    fig.update_layout(
        title="DP privacy cost (federated error ≈ 0% without DP)",
        template="plotly_white",
        height=480,
    )
    return _save(fig, out, "C2_analytics_fed_vs_dp_error")


def chart_analytics_latency(live_no_dp: list[dict], live_dp: list[dict], out: Path) -> list[str]:
    lat = [r for r in live_no_dp if r["metric_name"] in LATENCY_METRICS and not r["suppressed"]]
    if not lat:
        return []
    row = lat[0]
    tv, fv = float(row["true_value"]), float(row["federated_value"])
    dp_row = next(
        (r for r in live_dp if r["metric_name"] in LATENCY_METRICS and not r.get("feature")),
        row,
    )
    dv = dp_row.get("dp_noisy_value")
    dv = float(dv) if dv is not None and pd.notna(dv) else tv
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=["True (central)", "DP noisy"],
        y=[tv, dv],
        marker_color=["#27ae60", "#c0392b"],
        text=[f"{tv:.0f} ms", f"{dv:.0f} ms"],
        textposition="outside",
    ))
    subtitle = "Federated ≡ True" if abs(tv - fv) < 0.01 else f"Federated {fv:.0f} ms"
    fig.update_layout(
        title=f"Avg latency — {subtitle}; DP err {_rel_err(tv, dv)*100:.1f}%",
        yaxis_title="Milliseconds (inter-event gaps, capped 5000ms/event)",
        template="plotly_white",
        height=440,
    )
    fig.update_yaxes(range=_axis_range([tv, dv], floor=0, cap=None, pad_frac=0.15))
    return _save(fig, out, "D_analytics_latency_ms")


def chart_analytics_error(live_dp: list[dict], out: Path) -> list[str]:
    sub = pd.DataFrame([
        r for r in live_dp
        if not r["suppressed"]
        and not r.get("feature")
        and r["metric_name"] in DP_ERROR_METRICS
    ])
    if sub.empty:
        return []
    rate_rows, lat_row = [], None
    for _, r in sub.iterrows():
        tv = float(r["true_value"])
        dv = r["dp_noisy_value"]
        if dv is None or pd.isna(dv):
            continue
        err = _rel_err(tv, float(dv))
        if r["metric_name"] in RATE_METRICS:
            rate_rows.append({"label": metric_display_name(r["metric_name"]), "dp_err": err})
        elif r["metric_name"] in LATENCY_METRICS:
            lat_row = {"true_ms": tv, "dp_ms": float(dv), "dp_err": err}
    if not rate_rows and not lat_row:
        return []
    fig = make_subplots(
        rows=1, cols=2 if lat_row else 1,
        subplot_titles=(
            ["Rate metrics: DP error vs true", "Latency: true vs DP (ms)"]
            if lat_row
            else ["Rate metrics: DP error vs true"]
        ),
    )
    if rate_rows:
        rdf = pd.DataFrame(rate_rows)
        fig.add_trace(
            go.Bar(
                x=rdf["label"], y=rdf["dp_err"], marker_color="#c0392b",
                text=[f"{e:.1%}" for e in rdf["dp_err"]], textposition="outside",
            ),
            row=1, col=1,
        )
        fig.update_yaxes(tickformat=".1%", title_text="Relative error", row=1, col=1)
    if lat_row:
        fig.add_trace(
            go.Bar(
                x=["True", "DP noisy"],
                y=[lat_row["true_ms"], lat_row["dp_ms"]],
                marker_color=["#27ae60", "#c0392b"],
                text=[f"{lat_row['true_ms']:.0f}", f"{lat_row['dp_ms']:.0f}"],
                textposition="outside",
                showlegend=False,
            ),
            row=1, col=2,
        )
        fig.add_annotation(
            text=f"DP err {lat_row['dp_err']:.1%}",
            xref="x2 domain", yref="y2 domain", x=0.5, y=1.15, showarrow=False,
        )
        fig.update_yaxes(title_text="ms", row=1, col=2)
    fig.update_layout(
        title="DP privacy cost (rates + latency only; cohort/histogram excluded)",
        template="plotly_white",
        height=460,
        showlegend=False,
    )
    fig.update_xaxes(tickangle=20, row=1, col=1)
    return _save(fig, out, "F_analytics_dp_error")


def chart_histogram_buckets(meta: dict, out: Path) -> list[str]:
    hist = meta.get("histogram") or meta.get("true_histogram") or {}
    pct = histogram_percentages(hist)
    counts = [hist.get(b, 0) for b in LATENCY_BUCKET_ORDER]
    total = sum(counts)
    empty = meta.get("empty_buckets") or [b for b in LATENCY_BUCKET_ORDER if counts[LATENCY_BUCKET_ORDER.index(b)] == 0]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=LATENCY_BUCKET_ORDER,
        y=[pct[b] for b in LATENCY_BUCKET_ORDER],
        marker_color=["#bdc3c7" if c == 0 else "#3498db" for c in counts],
        text=[f"{pct[b]:.1f}%\n(n={c})" for b, c in zip(LATENCY_BUCKET_ORDER, counts)],
        textposition="outside",
    ))
    empty_note = f" Empty buckets: {', '.join(empty)}." if empty else ""
    max_ev = meta.get("max_events_per_client", "?")
    fig.update_layout(
        title=(
            f"Bounded analytics round ({total} events, ≤{max_ev}/client) — "
            f"5000+ = capped inter-arrival gaps.{empty_note}"
        ),
        yaxis_title="Share of events (%)",
        xaxis_title="Latency bucket (ms)",
        template="plotly_white",
        height=500,
    )
    ymax = max(pct.values()) * 1.25 + 5 if any(pct.values()) else 10
    fig.update_yaxes(range=[0, ymax])
    return _save(fig, out, "E_latency_histogram_buckets")


def chart_client_scale(out: Path) -> list[str]:
    from app.simulation.real_telemetry_loader import get_real_data_summary
    from app.simulation.synthetic_telemetry import generate_synthetic_clients

    comp = get_real_data_summary(partition_by="component")
    pid = get_real_data_summary(partition_by="pid")
    syn = generate_synthetic_clients(num_clients=200, seed=42)
    syn_events = sum(len(c.records) for c in syn)
    df = pd.DataFrame([
        {"deployment": f"Real component ({comp['num_clients']})", "clients": comp["num_clients"], "events": comp["total_events"]},
        {"deployment": f"Real pid ({pid['num_clients']})", "clients": pid["num_clients"], "events": pid["total_events"]},
        {"deployment": "Synthetic (200)", "clients": 200, "events": syn_events},
    ])
    df["events_per_client"] = df["events"] / df["clients"].clip(lower=1)
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Federated clients", "Events per client"))
    fig.add_trace(
        go.Bar(
            x=df["deployment"], y=df["clients"], marker_color="#3498db",
            text=df["clients"], textposition="outside", showlegend=False,
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Bar(
            x=df["deployment"], y=df["events_per_client"], marker_color="#9b59b6",
            text=[f"{v:.0f}" for v in df["events_per_client"]], textposition="outside", showlegend=False,
        ),
        row=1, col=2,
    )
    fig.update_xaxes(tickangle=25)
    fig.update_yaxes(title_text="Count", row=1, col=1)
    fig.update_yaxes(title_text="Mean events / client", row=1, col=2)
    fig.update_layout(
        title="Deployment scale — pid partition has more clients but sparser events/client",
        template="plotly_white",
        height=460,
    )
    return _save(fig, out, "K_client_scale_comparison")


def chart_real_data(out: Path) -> list[str]:
    from app.simulation.real_telemetry_loader import load_real_hdfs_clients

    clients = load_real_hdfs_clients(partition_by="component")
    rows = [
        {
            "client": c.client_id,
            "subsystem": c.client_id.replace("hdfs_", ""),
            "feature": r.feature,
            "latency_ms": r.latency_ms,
            "failure": r.failure,
        }
        for c in clients
        for r in c.records
    ]
    if not rows:
        return []
    df = pd.DataFrame(rows)
    full_hist = _bucket_counts(df["latency_ms"].tolist())
    full_pct = histogram_percentages(full_hist)
    fig1 = go.Figure()
    fig1.add_trace(go.Bar(
        x=LATENCY_BUCKET_ORDER,
        y=[full_pct[b] for b in LATENCY_BUCKET_ORDER],
        marker_color="#16a085",
        text=[f"{full_pct[b]:.1f}%\n(n={full_hist[b]})" for b in LATENCY_BUCKET_ORDER],
        textposition="outside",
    ))
    empty_full = [b for b in LATENCY_BUCKET_ORDER if full_hist.get(b, 0) == 0]
    note = f" Empty: {', '.join(empty_full)}." if empty_full else ""
    fig1.update_layout(
        title=f"Full HDFS dataset latency buckets ({len(df)} events, 5 subsystems).{note}",
        yaxis_title="% of events",
        xaxis_title="Inter-event latency bucket (ms)",
        template="plotly_white",
        height=480,
    )
    ymax = max(full_pct.values()) * 1.25 + 5 if any(full_pct.values()) else 10
    fig1.update_yaxes(range=[0, ymax])
    p1 = _save(fig1, out, "G_real_latency_distribution")

    fail = df.groupby("subsystem")["failure"].mean().reset_index(name="failure_rate")
    ymax = max(float(fail["failure_rate"].max()) * 1.35, 0.02)
    colors = [
        "#e74c3c" if v > 0 else "#bdc3c7" for v in fail["failure_rate"]
    ]
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(
        x=fail["subsystem"],
        y=fail["failure_rate"],
        marker_color=colors,
        text=[
            f"{v:.1%}" if v > 0 else "0% (none in sample)"
            for v in fail["failure_rate"]
        ],
        textposition="outside",
    ))
    fig2.update_layout(
        title=(
            "Failure rate by HDFS subsystem — WARN/ERROR concentrated in DataXceiver "
            "(0% = no failures in this subsystem slice)"
        ),
        yaxis_tickformat=".1%",
        yaxis_range=[0, ymax],
        template="plotly_white",
        height=420,
    )
    fig2.update_xaxes(tickangle=25)
    p2 = _save(fig2, out, "H_failure_rate_by_feature")

    ev = df.groupby("client").size().reset_index(name="events")
    ev["label"] = ev["client"].str.replace("hdfs_", "", regex=False)
    fig3 = px.bar(ev, x="label", y="events", title="Events per federated client (HDFS subsystem)")
    fig3.update_xaxes(tickangle=35)
    fig3.update_layout(height=420, yaxis_title="Event count")
    p3 = _save(fig3, out, "I_events_per_client")
    return p1 + p2 + p3


def build_index(out_dir: Path, files: list[str], notes: str) -> None:
    items = "\n".join(f'<li><a href="{f}">{f}</a></li>' for f in sorted(set(files)) if f.endswith(".html"))
    (out_dir / "index.html").write_text(
        f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>FedPrivacyLab</title></head><body>
<h1>FedPrivacyLab Graph Gallery</h1><p>{notes}</p>
<p>Generated {datetime.now(timezone.utc).isoformat()}</p><ul>{items}</ul></body></html>""",
        encoding="utf-8",
    )


def _clear_dir(path: Path) -> None:
    import shutil
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def main():
    _clear_dir(LATEST_DIR)
    out_dir = LATEST_DIR
    print(f"Generating graphs -> {out_dir}")

    tables = load_db()
    exp = tables.get("experiments", pd.DataFrame())
    metrics = tables.get("round_metrics", pd.DataFrame())
    files: list[str] = []

    for ds, a_stem, b_stem, f_stem, j_stem in [
        ("real_hdfs_loghub", "A_learning_real_hdfs", "B_training_real_hdfs", "F_privacy_real_hdfs", "J_fedavg_real_hdfs"),
        ("real_hdfs_loghub_pid", "A_learning_real_hdfs_pid", "B_training_real_hdfs_pid", "F_privacy_real_hdfs_pid", "J_fedavg_real_hdfs_pid"),
        ("synthetic_telemetry", "A_learning_synthetic", "B_training_synthetic", "F_privacy_synthetic", "J_fedavg_synthetic"),
    ]:
        files.extend(chart_learning_modes(exp, metrics, out_dir, ds, a_stem))
        files.extend(chart_training_curves(exp, metrics, out_dir, ds, b_stem))
        files.extend(chart_privacy_utility(exp, metrics, out_dir, ds, f_stem))
        files.extend(chart_fedavg_improvement(exp, metrics, out_dir, ds, j_stem))

    live_no_dp, live_meta, n_clients = [], {}, 0
    live_dp: list[dict] = []
    try:
        live_no_dp, live_meta, n_clients = live_real_analytics(dp_enabled=False)
        live_dp, _, _ = live_real_analytics(dp_enabled=True, epsilon=2.0)
    except Exception as exc:
        print(f"Warning: analytics skipped: {exc}")

    if live_no_dp and live_dp:
        files.extend(chart_analytics_rates(live_no_dp, live_dp, out_dir))
        files.extend(chart_analytics_true_fed_dp(live_no_dp, live_dp, out_dir))
        files.extend(chart_analytics_latency(live_no_dp, live_dp, out_dir))
        files.extend(chart_analytics_error(live_dp, out_dir))
        files.extend(chart_histogram_buckets(live_meta, out_dir))

    files.extend(chart_client_scale(out_dir))
    files.extend(chart_real_data(out_dir))

    notes = (
        f"Analytics: live HDFS ({n_clients} subsystem clients). "
        "Learning: final-round holdout metrics; synthetic 200-client + real HDFS. "
        "See data/results/DOCUMENTATION_ALIGNMENT.md."
    )
    build_index(out_dir, files, notes)
    batch_note = ""
    if not exp.empty and not metrics.empty:
        for ds in ("synthetic_telemetry", "real_hdfs_loghub", "real_hdfs_loghub_pid"):
            b = latest_comparison_experiments(exp, ds, metrics)
            if not b.empty:
                ids = b[b["mode"].isin(LEARNING_MODES)]["id"].tolist()
                batch_note += f" {ds} exp_ids={ids};"
    meta = {
        "output_dir": str(out_dir),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "analytics_clients": n_clients,
        "comparison_batches": batch_note.strip(),
        "files": sorted(set(files)),
        "notes": notes,
    }
    (out_dir / "manifest.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Done. {len(set(files))} files.")


if __name__ == "__main__":
    main()
