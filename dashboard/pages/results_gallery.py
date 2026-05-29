"""Results Gallery — canonical charts from graphs/latest."""

import json
import subprocess
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent.parent
LATEST_GRAPHS = ROOT / "data" / "results" / "graphs" / "latest"
LATEST_GRAD = ROOT / "data" / "results" / "graduate_report" / "latest"

st.title("Results Gallery")
st.markdown("Canonical visualizations: **`data/results/graphs/latest/`**")

if st.button("Regenerate graphs", type="primary"):
    with st.spinner("Running generate_graphs.py..."):
        r = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "generate_graphs.py")],
            capture_output=True,
            text=True,
            cwd=str(ROOT),
        )
    if r.returncode == 0:
        st.success("Done.")
        st.code(r.stdout or "OK")
    else:
        st.error(r.stderr or r.stdout)
    st.rerun()

if not LATEST_GRAPHS.exists():
    st.warning("No charts yet. Run: `python scripts/generate_graphs.py`")
    st.stop()

manifest = LATEST_GRAPHS / "manifest.json"
if manifest.exists():
    meta = json.loads(manifest.read_text(encoding="utf-8"))
    st.caption(f"Generated: {meta.get('generated_at', '—')} | Analytics exp: {meta.get('analytics_experiment_id')}")

md = LATEST_GRAD / "GRADUATE_RESULTS.md"
if md.exists():
    st.markdown(md.read_text(encoding="utf-8"))

st.subheader("Charts")
for hf in sorted(LATEST_GRAPHS.glob("*.html")):
    if hf.name == "index.html":
        continue
    with st.expander(hf.stem.replace("_", " ").title(), expanded=hf.name.startswith("A_")):
        st.components.v1.html(hf.read_text(encoding="utf-8"), height=520, scrolling=True)

idx = LATEST_GRAPHS / "index.html"
if idx.exists():
    st.link_button("Open gallery index in browser", f"file:///{idx.as_posix()}")

csv = LATEST_GRAPHS / "comparison_summary.csv"
if not csv.exists() and (LATEST_GRAD / "comparison_summary.csv").exists():
    csv = LATEST_GRAD / "comparison_summary.csv"
if csv.exists():
    st.subheader("Numeric summary")
    st.dataframe(__import__("pandas").read_csv(csv), use_container_width=True)
