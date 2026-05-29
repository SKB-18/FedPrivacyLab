"""Privacy risk report dashboard."""

import streamlit as st

from app.simulation.attack_simulator import describe_attack_risks
from dashboard.db_utils import get_experiment, get_experiment_ids, load_table, parse_privacy_notes

st.title("Risk Report")
st.markdown("Threat model, mitigations, and experiment-specific risk assessment.")

exp_ids = get_experiment_ids()
exp_id = st.selectbox("Experiment ID", exp_ids if exp_ids else [None], key="risk_exp")

if exp_id and exp_ids:
    exp = get_experiment(exp_id)
    if exp:
        mode = exp.get("mode", "")
        risks = [
            {
                "area": "Raw data upload",
                "risk": "High" if mode == "centralized_baseline" else "Low",
                "mitigation": "Federated local storage; no raw logs on server",
            },
            {
                "area": "Update leakage",
                "risk": "High" if mode == "fedavg" else "Medium" if "secureagg" in mode else "Low",
                "mitigation": "Secure aggregation simulation, clipping, DP noise",
            },
            {
                "area": "Dropout",
                "risk": "Medium",
                "mitigation": f"Simulated dropout rate: {exp.get('dropout_rate', 0)}",
            },
            {
                "area": "Poisoning / non-IID",
                "risk": "Medium",
                "mitigation": "Update clipping, contribution bounds, cohort suppression",
            },
        ]
        st.table(__import__("pandas").DataFrame(risks))

reports = load_table("privacy_reports")
if exp_id and not reports.empty:
    rsub = reports[reports["experiment_id"] == exp_id].tail(1)
    if not rsub.empty:
        row = rsub.iloc[0]
        st.subheader(f"Risk Level: {row.get('risk_level', 'unknown')}")
        for note in parse_privacy_notes(row.get("notes")):
            st.write(f"- {note}")

st.subheader("Attack Surface Reference")
for item in describe_attack_risks():
    with st.expander(item["risk"]):
        st.write(item["description"])
        st.success(f"Mitigation: {item['mitigation']}")
