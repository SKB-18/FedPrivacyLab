"""Privacy-utility tradeoff visualization."""

import streamlit as st
import plotly.express as px
import numpy as np

from dashboard.db_utils import get_experiment_ids, load_table

st.title("Utility Tradeoff")
st.markdown("Privacy strength vs model utility and analytics error.")

exp_ids = get_experiment_ids()
if not exp_ids:
    st.warning("No experiments found.")
    st.stop()

exp_id = st.selectbox("Experiment ID", exp_ids, key="utility_exp")

metrics_df = load_table("round_metrics")
analytics_df = load_table("analytics_results")
exps_df = load_table("experiments")

if not exps_df.empty:
    st.subheader("Cross-Experiment Comparison")
    fig_data = []
    for _, row in exps_df.iterrows():
        eid = row["id"]
        msub = metrics_df[metrics_df["experiment_id"] == eid] if not metrics_df.empty else None
        asub = analytics_df[analytics_df["experiment_id"] == eid] if not analytics_df.empty else None
        acc = msub["accuracy"].max() if msub is not None and not msub.empty else None
        rel_err = asub["relative_error"].mean() if asub is not None and not asub.empty else None
        fig_data.append(
            {
                "mode": row["mode"],
                "epsilon": row.get("epsilon"),
                "accuracy": acc,
                "mean_relative_error": rel_err,
                "dp_enabled": row.get("dp_enabled"),
            }
        )
    comp = __import__("pandas").DataFrame(fig_data)
    if comp["accuracy"].notna().any():
        fig = px.scatter(
            comp,
            x="epsilon",
            y="accuracy",
            color="mode",
            size_max=15,
            title="Epsilon vs Accuracy by Experiment Mode",
        )
        st.plotly_chart(fig, use_container_width=True)
    if comp["mean_relative_error"].notna().any():
        fig2 = px.bar(
            comp.dropna(subset=["mean_relative_error"]),
            x="mode",
            y="mean_relative_error",
            title="Analytics Relative Error by Mode",
            color="dp_enabled",
        )
        st.plotly_chart(fig2, use_container_width=True)

msub = metrics_df[metrics_df["experiment_id"] == exp_id] if not metrics_df.empty else metrics_df
if msub is not None and not msub.empty:
    st.subheader("ROC-AUC vs Round")
    fig3 = px.line(msub, x="round_number", y="roc_auc", markers=True)
    st.plotly_chart(fig3, use_container_width=True)

asub = analytics_df[analytics_df["experiment_id"] == exp_id] if not analytics_df.empty else analytics_df
if asub is not None and not asub.empty and "epsilon" in asub.columns:
    st.subheader("Epsilon vs Relative Error (Analytics)")
    valid = asub.dropna(subset=["relative_error", "epsilon"])
    if not valid.empty:
        fig4 = px.scatter(valid, x="epsilon", y="relative_error", color="metric_name")
        st.plotly_chart(fig4, use_container_width=True)
