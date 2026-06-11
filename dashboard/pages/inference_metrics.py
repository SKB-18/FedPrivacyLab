"""
Inference Metrics dashboard page.

Polls the inference coordinator (port 8001) every 5 seconds and displays:
  - KPI cards: total inferences, avg latency, validation accuracy, epsilon spent
  - Inference volume over time (cumulative)
  - Latency percentiles (p50, p95, p99)
  - Inference latency by client (bar chart)
  - Model accuracy drift (validation accuracy vs model version)
  - Privacy epsilon consumption (cumulative + burn rate)
  - Active clients table
  - Alerts for accuracy drift > 5% or epsilon < 10%
"""

from __future__ import annotations

import os
import time
from datetime import datetime

import numpy as np
import streamlit as st
import plotly.graph_objects as go

COORDINATOR_URL = os.environ.get("INFERENCE_COORDINATOR_URL", "http://localhost:8001")
DRIFT_ALERT_PCT = 5.0   # alert if accuracy drops more than 5%
EPSILON_ALERT_PCT = 0.1  # alert if avg epsilon remaining < 10% of budget


def _fetch(path: str) -> dict | None:
    """HTTP GET from the inference coordinator; returns None on failure."""
    try:
        if _try_requests(path) is not None:
            return _try_requests(path)
    except Exception:
        pass
    try:
        import urllib.request
        import json
        url = f"{COORDINATOR_URL}{path}"
        with urllib.request.urlopen(url, timeout=3) as r:
            return json.loads(r.read())
    except Exception:
        return None


def _try_requests(path: str) -> dict | None:
    try:
        import requests
        resp = requests.get(f"{COORDINATOR_URL}{path}", timeout=3)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        return None
    return None


def render() -> None:
    st.header("Inference Metrics")

    st.markdown(
        f"""
        Live monitoring of the **Federated Inference Coordinator** at `{COORDINATOR_URL}`.
        All predictions are made **on-device** — only aggregate metadata is reported here.
        """
    )

    # Auto-refresh toggle
    col_refresh, col_interval, _ = st.columns([2, 2, 6])
    with col_refresh:
        auto_refresh = st.toggle("Auto-refresh (5s)", value=False)
    with col_interval:
        if auto_refresh:
            st.caption("Polling every 5 seconds")

    # Fetch data from coordinator
    stats = _fetch("/inference_stats")
    clients_data = _fetch("/clients")
    drift_data = _fetch("/model_accuracy_drift")
    timeseries_data = _fetch("/inference_timeseries")

    coordinator_online = stats is not None

    if not coordinator_online:
        st.warning(
            f"Inference coordinator at `{COORDINATOR_URL}` is not reachable. "
            "Start it with: `uvicorn fedprivacylab.inference.coordinator:app --port 8001`"
        )
        _show_demo_dashboard()
        if auto_refresh:
            time.sleep(5)
            st.rerun()
        return

    # ── Alerts ──────────────────────────────────────────────────────────────────
    if drift_data:
        drift_pct = drift_data.get("drift_percent", 0.0)
        if drift_pct > DRIFT_ALERT_PCT:
            st.error(f"ALERT: Model accuracy drift is {drift_pct:.1f}% — exceeds {DRIFT_ALERT_PCT}% threshold!")

    clients_list = (clients_data or {}).get("clients", [])
    if clients_list:
        budget = 1.0  # Fetch from coordinator if available
        avg_eps_remaining = np.mean([c.get("epsilon_remaining", 1.0) for c in clients_list])
        if avg_eps_remaining < EPSILON_ALERT_PCT * budget:
            st.warning(f"ALERT: Average epsilon remaining is {avg_eps_remaining:.3f} — below 10% threshold!")

    # ── KPI cards ────────────────────────────────────────────────────────────────
    st.subheader("Key Performance Indicators")
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)

    total_preds = stats.get("total_predictions", 0)
    avg_latency = stats.get("avg_latency_ms", 0.0)
    val_acc = (drift_data or {}).get("current_accuracy", 0.0) if drift_data else 0.0
    eps_spent = stats.get("total_epsilon_spent", 0.0)

    kpi1.metric("Total Inferences Served", f"{total_preds:,}")
    kpi2.metric("Avg Latency", f"{avg_latency:.1f} ms")
    kpi3.metric("Validation Accuracy", f"{val_acc:.3f}")
    kpi4.metric("Total ε Spent", f"{eps_spent:.4f}")

    # ── Volume + latency charts ──────────────────────────────────────────────────
    st.subheader("Inference Volume Over Time")

    series = (timeseries_data or {}).get("series", [])

    if series:
        import pandas as pd
        df_ts = pd.DataFrame(series)
        df_ts["timestamp"] = pd.to_datetime(df_ts["timestamp"])
        df_ts = df_ts.sort_values("timestamp")
        df_ts["cumulative"] = df_ts["num_predictions"].cumsum()

        fig_vol = go.Figure(go.Scatter(
            x=df_ts["timestamp"], y=df_ts["cumulative"],
            fill="tozeroy", line=dict(color="#636EFA"), name="Cumulative inferences",
        ))
        fig_vol.update_layout(
            xaxis_title="Time", yaxis_title="Cumulative Inferences",
            template="plotly_dark", height=280,
        )
        st.plotly_chart(fig_vol, use_container_width=True)

        # Latency percentiles over time
        st.subheader("Latency Percentiles")
        latencies = df_ts["latency_ms"].values
        if len(latencies) >= 3:
            windows = np.array_split(latencies, min(len(latencies), 20))
            p50 = [np.percentile(w, 50) for w in windows]
            p95 = [np.percentile(w, 95) for w in windows]
            p99 = [np.percentile(w, 99) for w in windows]
            x_range = list(range(len(p50)))
            fig_lat = go.Figure()
            fig_lat.add_trace(go.Scatter(x=x_range, y=p50, name="p50", line=dict(color="#00CC96")))
            fig_lat.add_trace(go.Scatter(x=x_range, y=p95, name="p95", line=dict(color="#FFA15A")))
            fig_lat.add_trace(go.Scatter(x=x_range, y=p99, name="p99", line=dict(color="#EF553B")))
            fig_lat.update_layout(
                xaxis_title="Window", yaxis_title="Latency (ms)",
                template="plotly_dark", height=260,
            )
            st.plotly_chart(fig_lat, use_container_width=True)

        # Per-client latency bar chart
        st.subheader("Avg Latency by Client")
        by_client = df_ts.groupby("client_id")["latency_ms"].mean().reset_index()
        fig_client_lat = go.Figure(go.Bar(
            x=by_client["client_id"], y=by_client["latency_ms"],
            marker_color="#636EFA",
        ))
        fig_client_lat.update_layout(
            xaxis_title="Client", yaxis_title="Avg Latency (ms)",
            template="plotly_dark", height=260,
        )
        st.plotly_chart(fig_client_lat, use_container_width=True)

    else:
        st.info("No inference time-series data yet. Clients must log at least one batch.")

    # ── Accuracy drift ───────────────────────────────────────────────────────────
    st.subheader("Model Accuracy Drift")
    if drift_data and drift_data.get("drift_series"):
        drift_series = drift_data["drift_series"]
        versions = [d["model_version"] for d in drift_series]
        accuracies = [d["validation_accuracy"] for d in drift_series]
        fig_drift = go.Figure(go.Scatter(
            x=versions, y=accuracies, mode="lines+markers",
            line=dict(color="#AB63FA"), name="Validation Accuracy",
        ))
        baseline = drift_data.get("baseline_accuracy", accuracies[0] if accuracies else 0)
        fig_drift.add_hline(y=baseline, line_dash="dash", line_color="#FFA15A", annotation_text="Baseline")
        fig_drift.update_layout(
            xaxis_title="Model Version", yaxis_title="Validation Accuracy",
            template="plotly_dark", height=280,
        )
        st.plotly_chart(fig_drift, use_container_width=True)

    # ── Privacy epsilon ──────────────────────────────────────────────────────────
    st.subheader("Privacy ε Consumption")
    if clients_list:
        client_ids = [c["client_id"] for c in clients_list]
        eps_consumed = [c.get("epsilon_consumed", 0.0) for c in clients_list]
        eps_remaining = [c.get("epsilon_remaining", 1.0) for c in clients_list]

        fig_eps = go.Figure()
        fig_eps.add_trace(go.Bar(name="ε Consumed", x=client_ids, y=eps_consumed, marker_color="#EF553B"))
        fig_eps.add_trace(go.Bar(name="ε Remaining", x=client_ids, y=eps_remaining, marker_color="#00CC96"))
        fig_eps.update_layout(
            barmode="stack", xaxis_title="Client", yaxis_title="Epsilon",
            template="plotly_dark", height=280,
        )
        st.plotly_chart(fig_eps, use_container_width=True)

    # ── Active clients table ─────────────────────────────────────────────────────
    st.subheader("Active Clients")
    if clients_list:
        import pandas as pd
        df_clients = pd.DataFrame(clients_list)
        df_clients = df_clients.rename(columns={
            "client_id": "Client ID",
            "last_seen": "Last Seen",
            "total_inferences": "Total Inferences",
            "epsilon_consumed": "ε Consumed",
            "epsilon_remaining": "ε Remaining",
            "model_version": "Model Version",
        })
        st.dataframe(df_clients, use_container_width=True)
    else:
        st.info("No clients registered yet.")

    # Auto-refresh
    if auto_refresh:
        time.sleep(5)
        st.rerun()


def _show_demo_dashboard() -> None:
    """Render illustrative charts when the coordinator is offline."""
    st.subheader("Demo Data (coordinator offline)")

    rng = np.random.default_rng(42)
    n = 30
    t = list(range(n))

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Inferences", "4,350")
    col2.metric("Avg Latency", "23.4 ms")
    col3.metric("Validation Accuracy", "0.847")
    col4.metric("Total ε Spent", "0.1240")

    fig = go.Figure(go.Scatter(
        x=t, y=rng.integers(50, 200, n).cumsum(),
        fill="tozeroy", line=dict(color="#636EFA"),
    ))
    fig.update_layout(
        title="Cumulative Inferences (demo)", template="plotly_dark", height=250,
        xaxis_title="Time window", yaxis_title="Cumulative count",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Connect the inference coordinator to see live data.")


render()
