"""Federated analytics visualization — metrics split by unit."""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from dashboard.db_utils import get_experiment_ids, load_table
from app.analytics.metric_types import RATE_METRICS, LATENCY_METRICS, metric_display_name

st.title("Federated Analytics")
st.markdown("Rates, latency (ms), and errors on **separate charts** (correct units).")

exp_ids = get_experiment_ids()
if not exp_ids:
    st.warning("No experiments found.")
    st.stop()

exp_id = st.selectbox("Experiment ID", exp_ids, key="analytics_exp")
df = load_table("analytics_results")
sub = df[df["experiment_id"] == exp_id] if not df.empty else df
if sub.empty:
    st.info("No analytics results for this experiment.")
    st.stop()

global_rows = sub[(sub["feature"].isna()) | (sub["feature"] == "")]
if global_rows.empty:
    global_rows = sub

# --- Rates only ---
rate_df = global_rows[global_rows["metric_name"].isin(RATE_METRICS) & (~global_rows["suppressed"])]
if not rate_df.empty:
    st.subheader("Rates (0–100%)")
    plot_df = rate_df.groupby("metric_name", as_index=False).agg(
        true=("true_value", "mean"),
        federated=("federated_value", "mean"),
        dp=("dp_noisy_value", "mean"),
    )
    plot_df["label"] = plot_df["metric_name"].map(metric_display_name)
    fig = go.Figure()
    fig.add_trace(go.Bar(name="True", x=plot_df["label"], y=plot_df["true"], marker_color="#27ae60"))
    fig.add_trace(go.Bar(name="Federated", x=plot_df["label"], y=plot_df["federated"], marker_color="#2980b9"))
    if plot_df["dp"].notna().any():
        fig.add_trace(go.Bar(name="DP", x=plot_df["label"], y=plot_df["dp"], marker_color="#c0392b"))
    fig.update_layout(barmode="group", yaxis_tickformat=".0%", height=420)
    st.plotly_chart(fig, use_container_width=True)

# --- Latency only ---
lat_df = global_rows[global_rows["metric_name"].isin(LATENCY_METRICS) & (~global_rows["suppressed"])]
if not lat_df.empty:
    st.subheader("Average latency (milliseconds)")
    row = lat_df.iloc[0]
    fig2 = go.Figure(go.Bar(
        x=["True", "Federated", "DP"],
        y=[row["true_value"], row["federated_value"], row.get("dp_noisy_value") or row["federated_value"]],
        marker_color=["#27ae60", "#2980b9", "#c0392b"],
    ))
    fig2.update_layout(yaxis_title="ms", height=360)
    st.plotly_chart(fig2, use_container_width=True)

# --- Relative error ---
err_df = global_rows[~global_rows["suppressed"]].dropna(subset=["relative_error"])
if not err_df.empty:
    st.subheader("Relative error (federated vs true)")
    err_df = err_df.copy()
    err_df["label"] = err_df["metric_name"].map(metric_display_name)
    fig3 = px.bar(
        err_df.groupby("label", as_index=False)["relative_error"].mean(),
        x="label", y="relative_error",
        color="relative_error", color_continuous_scale="RdYlGn_r",
    )
    fig3.update_layout(showlegend=False, height=380)
    st.plotly_chart(fig3, use_container_width=True)

st.subheader("Detail table")
st.dataframe(
    global_rows[
        ["round_number", "metric_name", "true_value", "federated_value", "dp_noisy_value", "relative_error", "suppressed"]
    ],
    use_container_width=True,
)
