"""Train/eval split must not use a single tiny client on real HDFS scale."""

from app.config_loader import set_active_dataset
from app.coordinator.client_registry import (
    clear_registry,
    configure_train_eval_split,
    get_eval_meta,
    get_eval_records,
    populate_registry,
)


def test_real_hdfs_record_level_holdout():
    set_active_dataset("real_hdfs_loghub")
    eid = 42
    populate_registry(eid, num_clients=5, dataset="real_hdfs_loghub")
    configure_train_eval_split(eid, seed=42, dataset="real_hdfs_loghub")
    meta = get_eval_meta(eid)
    eval_recs = get_eval_records(eid)
    clear_registry(eid)
    assert meta["split"] == "record"
    assert len(eval_recs) >= 100
    assert meta["eval_positive_rate"] > 0.05


def test_real_hdfs_pid_uses_record_holdout():
    set_active_dataset("real_hdfs_loghub_pid")
    eid = 43
    populate_registry(eid, num_clients=13, dataset="real_hdfs_loghub_pid")
    configure_train_eval_split(eid, seed=42, dataset="real_hdfs_loghub_pid")
    meta = get_eval_meta(eid)
    eval_recs = get_eval_records(eid)
    clear_registry(eid)
    assert meta["split"] == "record"
    assert len(eval_recs) >= 100
    assert meta["eval_positive_rate"] >= 0.05
