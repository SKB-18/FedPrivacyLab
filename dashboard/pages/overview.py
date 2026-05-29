"""Overview dashboard tab."""

import streamlit as st
import plotly.express as px

from dashboard.db_utils import get_experiment, get_experiment_ids, load_table

st.title("Overview")
st.markdown("Experiment configuration, mode, and privacy status at a glance.")

exp_ids = get_experiment_ids()
if not exp_ids:
    st.warning("No experiments found. Start the API and run an experiment first.")
    st.code(
        "uvicorn app.main:app --reload\n"
        "curl -X POST http://127.0.0.1:8000/api/v1/experiments/start -H 'Content-Type: application/json' "
        "-d '{\"name\":\"demo\",\"mode\":\"fedavg_secureagg_dp\",\"num_clients\":100,\"rounds\":3}'"
    )
    st.stop()

exp_id = st.selectbox("Experiment ID", exp_ids)
exp = get_experiment(exp_id)

if exp:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mode", exp.get("mode", "—"))
    c2.metric("Clients", exp.get("num_clients", "—"))
    c3.metric("Rounds", exp.get("rounds", "—"))
    c4.metric("Status", exp.get("status", "—"))

    c5, c6, c7 = st.columns(3)
    c5.metric("DP Enabled", "Yes" if exp.get("dp_enabled") else "No")
    c6.metric("Secure Agg", "Yes" if exp.get("secure_agg_enabled") else "No")
    c7.metric("Dropout Rate", f"{(exp.get('dropout_rate') or 0)*100:.0f}%")

    raw_centralized = exp.get("mode") == "centralized_baseline"
    st.info(
        f"**Raw data centralized:** {'Yes' if raw_centralized else 'No'} — "
        f"Dataset: {exp.get('dataset', 'synthetic_telemetry')}"
    )

metrics_df = load_table("round_metrics")
if not metrics_df.empty and exp_id:
    sub = metrics_df[metrics_df["experiment_id"] == exp_id]
    if not sub.empty and "accuracy" in sub.columns:
        fig = px.line(
            sub,
            x="round_number",
            y="accuracy",
            title="Accuracy by Round",
            markers=True,
        )
        st.plotly_chart(fig, use_container_width=True)
