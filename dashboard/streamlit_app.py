"""FedPrivacyLab Streamlit dashboard."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import streamlit as st

st.set_page_config(
    page_title="FedPrivacyLab",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.sidebar.title("FedPrivacyLab")
st.sidebar.markdown(
    "**Federated Analytics & Privacy-Preserving ML Workbench**"
)
st.sidebar.markdown("---")

pages = {
    "Overview": "pages/overview.py",
    "Real Data Scenario": "pages/real_data_scenario.py",
    "Results Gallery": "pages/results_gallery.py",
    "Federated Analytics": "pages/federated_analytics.py",
    "Training Progress": "pages/training_progress.py",
    "Privacy Controls": "pages/privacy_controls.py",
    "Utility Tradeoff": "pages/utility_tradeoff.py",
    "Client Participation": "pages/client_participation.py",
    "Risk Report": "pages/risk_report.py",
    "System Explanation": "pages/system_explanation.py",
}

selection = st.sidebar.radio("Navigate", list(pages.keys()))

page_path = ROOT / "dashboard" / pages[selection]
with open(page_path, encoding="utf-8") as f:
    code = f.read()
exec(compile(code, str(page_path), "exec"), {"__file__": str(page_path)})
