"""Federated learning progress charts."""

import streamlit as st
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from dashboard.db_utils import get_experiment_ids, load_table

st.title("Federated Learning Progress")
st.markdown("Round-by-round accuracy, loss, and classification metrics.")

exp_ids = get_experiment_ids()
if not exp_ids:
    st.warning("No experiments found.")
    st.stop()

exp_id = st.selectbox("Experiment ID", exp_ids, key="train_exp")
df = load_table("round_metrics")
sub = df[df["experiment_id"] == exp_id] if not df.empty else df

if sub.empty:
    st.info("No training metrics. Run a federated learning experiment.")
    st.stop()

fig = make_subplots(rows=2, cols=2, subplot_titles=("Accuracy", "ROC-AUC", "Loss", "F1"))
fig.add_trace(go.Scatter(x=sub["round_number"], y=sub["accuracy"], name="Accuracy"), row=1, col=1)
fig.add_trace(go.Scatter(x=sub["round_number"], y=sub["roc_auc"], name="ROC-AUC"), row=1, col=2)
fig.add_trace(go.Scatter(x=sub["round_number"], y=sub["eval_loss"], name="Loss"), row=2, col=1)
fig.add_trace(go.Scatter(x=sub["round_number"], y=sub["f1_score"], name="F1"), row=2, col=2)
fig.update_layout(height=600, showlegend=False, title_text="Training Metrics")
st.plotly_chart(fig, use_container_width=True)

if "precision_score" in sub.columns:
    fig2 = px.line(
        sub,
        x="round_number",
        y=["precision_score", "recall_score"],
        title="Precision & Recall",
        markers=True,
    )
    st.plotly_chart(fig2, use_container_width=True)

part = load_table("client_participation")
if not part.empty:
    psub = part[part["experiment_id"] == exp_id]
    dropout = psub.groupby("round_number").apply(
        lambda g: (g["selected"] & ~g["completed"]).sum()
    )
    fig3 = px.bar(x=dropout.index, y=dropout.values, title="Dropped Clients per Round")
    st.plotly_chart(fig3, use_container_width=True)
