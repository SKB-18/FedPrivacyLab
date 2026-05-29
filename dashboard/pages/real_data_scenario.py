"""Real-life data scenario: LogHub HDFS production logs."""

import sys
from pathlib import Path

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

st.title("Real Data Scenario")
st.markdown(
    """
**Production distributed-systems telemetry** from [LogPAI LogHub HDFS](https://github.com/logpai/loghub)
(2,000 structured Hadoop events). Each HDFS component (NameNode, DataNode, etc.) is a **federated client**;
the server never receives raw logs—only bounded aggregates and model updates.
"""
)

with st.spinner("Loading real HDFS telemetry summary..."):
    try:
        from app.simulation.real_telemetry_loader import (
            ensure_hdfs_dataset,
            get_real_data_summary,
            load_real_hdfs_clients,
        )

        ensure_hdfs_dataset()
        summary = get_real_data_summary()
        clients = load_real_hdfs_clients(max_clients=50)
    except Exception as e:
        st.error(f"Could not load real data: {e}")
        st.code("python scripts/download_real_data.py")
        st.stop()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Federated Clients", summary["num_clients"])
c2.metric("Total Events", summary["total_events"])
c3.metric("Failure Rate", f"{summary['failure_rate']*100:.2f}%")
c4.metric("Avg Latency (ms)", f"{summary['avg_latency_ms']:.0f}")

st.info(summary["description"])

# Build chart data from clients
import pandas as pd

rows = []
for c in clients:
    for r in c.records:
        rows.append(
            {
                "client": c.client_id,
                "feature": r.feature,
                "latency_ms": r.latency_ms,
                "failure": r.failure,
                "profile": c.profile,
            }
        )
df = pd.DataFrame(rows)

col1, col2 = st.columns(2)
with col1:
    fig_lat = px.histogram(
        df,
        x="latency_ms",
        color="feature",
        nbins=30,
        title="Real Event Latency Distribution by Feature",
        labels={"latency_ms": "Inter-event latency (ms)"},
    )
    st.plotly_chart(fig_lat, use_container_width=True)

with col2:
    fail_by_feat = df.groupby("feature")["failure"].mean().reset_index()
    fail_by_feat.columns = ["feature", "failure_rate"]
    fig_fail = px.bar(
        fail_by_feat,
        x="feature",
        y="failure_rate",
        title="Failure Rate by Feature (Real HDFS)",
        color="failure_rate",
        color_continuous_scale="Reds",
    )
    st.plotly_chart(fig_fail, use_container_width=True)

client_events = df.groupby("client").size().reset_index(name="events")
fig_clients = px.bar(
    client_events,
    x="client",
    y="events",
    title="Non-IID Load: Events per Federated Client (Component)",
)
fig_clients.update_xaxes(tickangle=45)
st.plotly_chart(fig_clients, use_container_width=True)

st.subheader("Start experiment on real data")
st.code(
    """curl -X POST http://127.0.0.1:8000/api/v1/experiments/start -H "Content-Type: application/json" -d '{
  "name": "real_hdfs_analytics",
  "mode": "federated_analytics",
  "dataset": "real_hdfs_loghub",
  "num_clients": 150,
  "rounds": 5,
  "clients_per_round": 100,
  "dp_enabled": true
}'""",
    language="json",
)
