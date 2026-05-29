"""Reverify data/results artifacts, DB experiments, and live analytics."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
from sqlalchemy import create_engine

GRAPH_DIR = ROOT / "data" / "results" / "graphs" / "latest"
GRAD_DIR = ROOT / "data" / "results" / "graduate_report" / "latest"
DB_PATH = ROOT / "data" / "fedprivacylab.db"


def main() -> int:
    issues: list[str] = []
    warnings: list[str] = []

    # Graph manifest vs disk
    manifest_path = GRAPH_DIR / "manifest.json"
    if not manifest_path.exists():
        issues.append("Missing graphs/latest/manifest.json")
    else:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        listed = set(manifest.get("files", []))
        on_disk = {
            p.name
            for p in GRAPH_DIR.iterdir()
            if p.is_file() and p.name not in ("manifest.json",)
        }
        missing = listed - on_disk
        if missing:
            issues.append(f"Manifest files missing on disk: {sorted(missing)[:5]}...")
        html_count = len([f for f in listed if f.endswith(".html")])
        if html_count < 17:
            issues.append(f"Expected >=17 HTML charts, manifest has {html_count}")

    if not (GRAPH_DIR / "index.html").exists():
        issues.append("Missing graphs/latest/index.html")

    # Graduate report
    for name in ("GRADUATE_RESULTS.md", "comparison_summary.csv"):
        if not (GRAD_DIR / name).exists():
            issues.append(f"Missing graduate_report/latest/{name}")

    # DB experiments
    if not DB_PATH.exists():
        issues.append("Missing fedprivacylab.db")
    else:
        engine = create_engine(f"sqlite:///{DB_PATH}")
        exp = pd.read_sql_table("experiments", engine)
        metrics = pd.read_sql_table("round_metrics", engine)
        for ds in ("real_hdfs_loghub", "synthetic_telemetry"):
            sub = exp[
                exp["name"].astype(str).str.startswith("comparison_")
                & (exp["dataset"].astype(str) == ds)
            ].sort_values("id").groupby("mode").tail(1)
            if sub.empty:
                issues.append(f"No comparison_* experiments for {ds}")
                continue
            print(f"\n{ds} — latest comparison per mode:")
            for _, row in sub.iterrows():
                m = metrics[metrics["experiment_id"] == row["id"]]
                best_f1 = float(m["f1_score"].max()) if not m.empty and m["f1_score"].notna().any() else None
                last_acc = float(m["accuracy"].iloc[-1]) if not m.empty else None
                print(f"  {row['mode']:24} id={int(row['id']):3}  acc={last_acc}  best_f1={best_f1}")
                if row["mode"] in ("fedavg", "centralized_baseline") and m.empty:
                    issues.append(f"{ds}/{row['mode']}: no round_metrics")
            if ds == "synthetic_telemetry":
                fed = sub[sub["mode"] == "fedavg"]
                if not fed.empty:
                    bf = metrics[metrics["experiment_id"] == fed.iloc[0]["id"]]["f1_score"].max()
                    if bf is not None and float(bf) <= 0.01:
                        warnings.append("Synthetic FedAvg best F1 still near zero")

    # Live analytics
    from scripts.generate_graphs import live_real_analytics

    no_dp, meta, n_clients = live_real_analytics(dp_enabled=False)
    dp, _, _ = live_real_analytics(dp_enabled=True, epsilon=2.0)
    if meta.get("true_histogram") != meta.get("histogram"):
        issues.append("true_histogram != histogram in live analytics meta")
    print(f"\nLive analytics: {n_clients} clients, {meta.get('events_in_round')} bounded events")
    print(f"  Empty buckets: {meta.get('empty_buckets')}")
    for r in no_dp:
        if r.get("feature") or r.get("suppressed"):
            continue
        if r["metric_name"] in ("count_failures_by_feature", "average_latency_by_feature", "click_rate_by_feature"):
            tv, fv = float(r["true_value"]), float(r["federated_value"])
            if abs(tv - fv) > 1e-6:
                issues.append(f"Fed != true for {r['metric_name']}: {tv} vs {fv}")
    dp_errs = []
    dp_map = {r["metric_name"]: r for r in dp if not r.get("feature")}
    for r in no_dp:
        if r.get("feature") or r.get("suppressed"):
            continue
        d = dp_map.get(r["metric_name"], {})
        dv = d.get("dp_noisy_value")
        if dv is not None and float(r["true_value"]) != 0:
            dp_errs.append(abs(float(dv) - float(r["true_value"])) / abs(float(r["true_value"])))
    if dp_errs and max(dp_errs) < 0.001:
        warnings.append("DP error very small — epsilon may be high for demo")
    print(f"  Mean DP relative error (rates+latency): {sum(dp_errs)/len(dp_errs):.2%}" if dp_errs else "  No DP errors computed")

    print("\n" + "=" * 60)
    print("REVERIFICATION SUMMARY")
    print("=" * 60)
    print(f"  Issues:   {len(issues)}")
    for i in issues:
        print(f"    [FAIL] {i}")
    print(f"  Warnings: {len(warnings)}")
    for w in warnings:
        print(f"    [WARN] {w}")
    if not issues:
        print("  Status: PASS — artifacts and logic consistent")
    else:
        print("  Status: FAIL — fix issues above")
    return 1 if issues else 0


if __name__ == "__main__":
    sys.exit(main())
