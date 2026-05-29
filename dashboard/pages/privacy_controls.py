"""Privacy controls dashboard."""

import streamlit as st

from dashboard.db_utils import get_experiment, get_experiment_ids, load_table, parse_privacy_notes

st.title("Privacy Controls")
st.markdown("Secure aggregation, DP parameters, clipping, and dropout configuration.")

exp_ids = get_experiment_ids()
if not exp_ids:
    st.warning("No experiments found.")
    st.stop()

exp_id = st.selectbox("Experiment ID", exp_ids, key="privacy_exp")
exp = get_experiment(exp_id)

if exp:
    st.subheader("Experiment Privacy Settings")
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Secure Aggregation:** {exp.get('secure_agg_enabled')}")
        st.write(f"**DP Enabled:** {exp.get('dp_enabled')}")
        st.write(f"**Epsilon:** {exp.get('epsilon')}")
        st.write(f"**Delta:** {exp.get('delta')}")
    with col2:
        st.write(f"**Clipping Norm:** {exp.get('clipping_norm')}")
        st.write(f"**Noise Multiplier:** {exp.get('noise_multiplier')}")
        st.write(f"**Dropout Rate:** {exp.get('dropout_rate')}")

reports = load_table("privacy_reports")
if not reports.empty:
    rsub = reports[reports["experiment_id"] == exp_id].tail(1)
    if not rsub.empty:
        row = rsub.iloc[0]
        st.subheader("Latest Privacy Report")
        st.metric("Risk Level", row.get("risk_level", "—"))
        notes = parse_privacy_notes(row.get("notes"))
        for note in notes:
            st.write(f"- {note}")
