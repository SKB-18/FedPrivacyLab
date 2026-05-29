"""Diagnose why real HDFS learning metrics hit 100%."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
from app.config_loader import set_active_dataset
from app.coordinator.client_registry import (
    clear_registry,
    configure_train_eval_split,
    get_eval_records,
    get_train_client_ids,
    populate_registry,
)
from app.learning.evaluator import evaluate_model
from app.learning.feature_encoding import fit_normalization, reset_normalization
from app.learning.model import build_model, records_to_xy
from app.simulation.real_telemetry_loader import load_real_hdfs_clients


def main():
    set_active_dataset("real_hdfs_loghub")
    clients = load_real_hdfs_clients(partition_by="component")
    print(f"Clients: {len(clients)}")
    for c in clients:
        recs = c.records
        pos = sum(r.label_poor_experience for r in recs)
        print(f"  {c.client_id}: n={len(recs)} poor_exp={pos} ({pos/len(recs):.1%}) failures={sum(r.failure for r in recs)}")

    clear_registry()
    populate_registry(999, num_clients=5, dataset="real_hdfs_loghub")
    n_train, n_eval = configure_train_eval_split(999, seed=42)
    eval_recs = get_eval_records(999)
    train_ids = get_train_client_ids(999)
    print(f"\nSplit: train_clients={n_train} eval_clients={n_eval} eval_records={len(eval_recs)}")
    y_eval = [int(r["label_poor_experience"]) for r in eval_recs]
    print(f"Eval labels: pos={sum(y_eval)} neg={len(y_eval)-sum(y_eval)} total={len(y_eval)}")
    if len(set(y_eval)) <= 1:
        print("WARNING: eval set is single-class -> accuracy/F1 can be trivially perfect")

    reset_normalization()
    train_recs = []
    from app.coordinator.client_registry import get_client_records_map
    for cid in train_ids:
        train_recs.extend(get_client_records_map(999).get(cid, []))
    fit_normalization(train_recs)
    X, y = records_to_xy(eval_recs)
    print(f"Eval arrays: X={X.shape} y positive rate={y.mean():.3f}")

    # Untrained model
    model = build_model()
    m0 = evaluate_model(model, X, y)
    print(f"Untrained eval: acc={m0['accuracy']:.3f} f1={m0['f1']:.3f}")

    clear_registry(999)


if __name__ == "__main__":
    main()
