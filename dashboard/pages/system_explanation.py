"""System explanation dashboard tab."""

import streamlit as st

st.title("System Explanation")
st.markdown(
    """
FedPrivacyLab is a **federated analytics and federated learning privacy workbench**.
Clients keep raw telemetry local; the server coordinates rounds and stores **aggregate-only** results.
"""
)

steps = [
    "Client data stays local — raw telemetry never leaves simulated clients.",
    "Server selects clients per round (with optional dropout simulation).",
    "Clients compute local analytics summaries or train local models.",
    "Each client contribution is bounded (analytics) or clipped (learning).",
    "Model updates are L2-clipped before aggregation.",
    "Secure aggregation is **simulated** via additive masks that cancel in the aggregate.",
    "Differential privacy noise can be added to aggregate analytics or model updates.",
    "Dashboard compares privacy strength with utility loss across experiment modes.",
]

st.subheader("Privacy-Preserving Federated Design")
for i, step in enumerate(steps, 1):
    st.markdown(f"{i}. {step}")

st.subheader("Experiment Modes")
modes = {
    "centralized_baseline": "Raw data centralized — upper bound on utility",
    "federated_analytics": "Bounded local summaries, optional DP",
    "fedavg": "FedAvg with visible individual updates",
    "fedavg_secureagg": "Masked updates; server sees aggregate only",
    "fedavg_secureagg_dp": "Secure agg + Gaussian DP on aggregate update",
}
for mode, desc in modes.items():
    st.write(f"**{mode}**: {desc}")

st.subheader("Limitations")
st.warning(
    "Secure aggregation is simulated, not production cryptography. "
    "Privacy accounting is simplified. Synthetic data may not reflect real user behavior."
)
