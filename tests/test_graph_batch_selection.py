"""Graph experiment batch selection must not mix comparison runs."""

import pandas as pd

from scripts.generate_graphs import latest_comparison_experiments


def test_latest_comparison_uses_same_batch_window():
    exp = pd.DataFrame([
        {"id": 94, "name": "comparison_centralized_baseline", "mode": "centralized_baseline",
         "dataset": "real_hdfs_loghub", "status": "completed"},
        {"id": 96, "name": "comparison_fedavg_plain", "mode": "fedavg",
         "dataset": "real_hdfs_loghub", "status": "completed"},
        {"id": 97, "name": "comparison_fedavg_secureagg", "mode": "fedavg_secureagg",
         "dataset": "real_hdfs_loghub", "status": "completed"},
        {"id": 171, "name": "validate_learning", "mode": "fedavg",
         "dataset": "real_hdfs_loghub", "status": "completed"},
    ])
    sub = latest_comparison_experiments(exp, "real_hdfs_loghub", metrics=None)
    fed = sub[sub["mode"] == "fedavg"]
    assert int(fed.iloc[0]["id"]) == 96
    assert 171 not in sub["id"].tolist()
