"""Client participation visualization."""

import streamlit as st
import plotly.express as px

from dashboard.db_utils import get_experiment_ids, load_table

st.title("Client Participation")
st.markdown("Selected, completed, and dropped clients; update norms and clipping.")

exp_ids = get_experiment_ids()
if not exp_ids:
    st.warning("No experiments found.")
    st.stop()

exp_id = st.selectbox("Experiment ID", exp_ids, key="part_exp")
df = load_table("client_participation")
sub = df[df["experiment_id"] == exp_id] if not df.empty else df

if sub.empty:
    st.info("No participation records.")
    st.stop()

summary = sub.groupby("round_number").agg(
    selected=("selected", "sum"),
    completed=("completed", "sum"),
    clipped=("clipped", "sum"),
).reset_index()
summary["dropped"] = summary["selected"] - summary["completed"]

fig = px.bar(
    summary,
    x="round_number",
    y=["selected", "completed", "dropped"],
    title="Client Participation per Round",
    barmode="group",
)
st.plotly_chart(fig, use_container_width=True)

if "update_norm" in sub.columns:
    norms = sub[sub["update_norm"].notna()]
    if not norms.empty:
        fig2 = px.histogram(norms, x="update_norm", color="clipped", title="Update Norm Distribution")
        st.plotly_chart(fig2, use_container_width=True)

clipped_pct = sub["clipped"].mean() * 100 if "clipped" in sub.columns else 0
st.metric("Clipped Updates %", f"{clipped_pct:.1f}%")
st.dataframe(sub.head(100), use_container_width=True)
