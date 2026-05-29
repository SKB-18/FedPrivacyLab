"""In-memory client registry (raw data stays local to clients)."""

from app.clients.telemetry_client import TelemetryClient
from app.simulation.real_telemetry_loader import load_real_hdfs_clients
from app.simulation.synthetic_telemetry import generate_synthetic_clients

_registry: dict[int, dict[str, TelemetryClient]] = {}
_train_client_ids: dict[int, list[str]] = {}
_train_records_map: dict[int, dict[str, list[dict]]] = {}
_eval_records: dict[int, list[dict]] = {}
_eval_meta: dict[int, dict] = {}


def populate_registry(
    experiment_id: int,
    num_clients: int,
    seed: int = 42,
    non_iid_severity: float = 1.0,
    min_records: int = 5,
    max_records: int = 50,
    dataset: str = "synthetic_telemetry",
) -> int:
    """Generate or load clients for an experiment. Raw records never leave client objects."""
    if dataset in ("real_hdfs_loghub", "real_hdfs_loghub_pid"):
        # component: 5 subsystems, ~2k events, failures (primary real-life demo).
        # pid: up to 13 process clients (scale chart / more federated clients).
        partition = "pid" if dataset == "real_hdfs_loghub_pid" else "component"
        synthetic = load_real_hdfs_clients(
            max_clients=num_clients,
            min_records_per_client=min_records,
            partition_by=partition,
        )
    else:
        synthetic = generate_synthetic_clients(
            num_clients=num_clients,
            min_records=min_records,
            max_records=max_records,
            seed=seed,
            non_iid_severity=non_iid_severity,
        )

    clients = {c.client_id: TelemetryClient(c) for c in synthetic}
    _registry[experiment_id] = clients
    _train_client_ids.pop(experiment_id, None)
    _train_records_map.pop(experiment_id, None)
    _eval_records.pop(experiment_id, None)
    _eval_meta.pop(experiment_id, None)
    return len(clients)


def _stratified_record_holdout(
    recs: list[dict],
    n_ev: int,
    rng,
) -> tuple[list[dict], list[dict]]:
    """Per-client record holdout keeping both classes when possible."""
    import numpy as np

    if not recs:
        return [], []
    n_ev = min(max(1, n_ev), len(recs))
    pos_idx = [i for i, r in enumerate(recs) if int(r.get("label_poor_experience", 0))]
    neg_idx = [i for i in range(len(recs)) if i not in pos_idx]
    if pos_idx and neg_idx and n_ev >= 2:
        n_pos_ev = max(1, round(n_ev * len(pos_idx) / len(recs)))
        n_pos_ev = min(n_pos_ev, len(pos_idx), n_ev - 1)
        n_neg_ev = min(len(neg_idx), n_ev - n_pos_ev)
        ev_idx = set(rng.choice(pos_idx, size=n_pos_ev, replace=False))
        ev_idx |= set(rng.choice(neg_idx, size=n_neg_ev, replace=False))
    else:
        ev_idx = set(rng.choice(len(recs), size=n_ev, replace=False))
    eval_recs = [recs[i] for i in sorted(ev_idx)]
    train_recs = [r for i, r in enumerate(recs) if i not in ev_idx]
    return train_recs, eval_recs


def configure_train_eval_split(
    experiment_id: int,
    seed: int = 42,
    eval_client_fraction: float = 0.2,
    dataset: str | None = None,
) -> tuple[int, int]:
    """
    Hold out ~20% of clients for evaluation only (never used in local training).
    Returns (num_train_clients, num_eval_clients).
    """
    import numpy as np

    ids = sorted(get_all_clients(experiment_id).keys())
    if not ids:
        _train_client_ids[experiment_id] = []
        _train_records_map[experiment_id] = {}
        _eval_records[experiment_id] = []
        _eval_meta[experiment_id] = {"split": "none"}
        return 0, 0

    full_map = {
        cid: client.get_local_records()
        for cid, client in get_all_clients(experiment_id).items()
    }
    rng = np.random.default_rng(seed)

    # Real HDFS (component or pid): always record holdout so eval has enough positives.
    # Client holdout on 13 pid clients leaves ~1 positive → F1=0 with high accuracy.
    # Few synthetic clients: same record holdout rule.
    use_record_holdout = len(ids) <= 10 or dataset in (
        "real_hdfs_loghub",
        "real_hdfs_loghub_pid",
    )
    if use_record_holdout:
        train_map: dict[str, list[dict]] = {}
        eval_list: list[dict] = []
        for cid in ids:
            recs = full_map.get(cid, [])
            if not recs:
                train_map[cid] = []
                continue
            n_ev = max(1, int(len(recs) * eval_client_fraction))
            train_part, eval_part = _stratified_record_holdout(recs, n_ev, rng)
            train_map[cid] = train_part
            eval_list.extend(eval_part)
        if dataset in ("real_hdfs_loghub", "real_hdfs_loghub_pid"):
            min_eval_pos = 10
            n_pos = sum(int(r.get("label_poor_experience", 0)) for r in eval_list)
            if n_pos < min_eval_pos:
                pool: list[tuple[str, dict]] = []
                for cid in ids:
                    for r in train_map.get(cid, []):
                        if int(r.get("label_poor_experience", 0)):
                            pool.append((cid, r))
                rng.shuffle(pool)
                for cid, r in pool[: min(min_eval_pos - n_pos, len(pool))]:
                    train_map[cid].remove(r)
                    eval_list.append(r)
        _train_client_ids[experiment_id] = ids
        _train_records_map[experiment_id] = train_map
        _eval_records[experiment_id] = eval_list
        pos = sum(int(r.get("label_poor_experience", 0)) for r in eval_list)
        _eval_meta[experiment_id] = {
            "split": "record",
            "num_eval_records": len(eval_list),
            "num_eval_clients": len(ids),
            "num_train_clients": len(ids),
            "eval_positive_rate": pos / max(len(eval_list), 1),
        }
        return len(ids), len(ids)

    n_eval = max(1, int(len(ids) * eval_client_fraction)) if len(ids) >= 3 else 1
    if len(ids) <= 2:
        n_eval = 1
    eval_ids = set(rng.choice(ids, size=min(n_eval, len(ids)), replace=False))
    train_ids = [cid for cid in ids if cid not in eval_ids]
    if not train_ids:
        train_ids = [ids[0]]
        eval_ids = set(ids[1:]) or eval_ids

    _train_client_ids[experiment_id] = train_ids
    _train_records_map[experiment_id] = {
        cid: full_map.get(cid, []) for cid in train_ids
    }
    eval_list = [r for cid in eval_ids for r in full_map.get(cid, [])]
    _eval_records[experiment_id] = eval_list
    pos = sum(int(r.get("label_poor_experience", 0)) for r in eval_list)
    _eval_meta[experiment_id] = {
        "split": "client",
        "num_eval_records": len(eval_list),
        "num_eval_clients": len(eval_ids),
        "num_train_clients": len(train_ids),
        "eval_positive_rate": pos / max(len(eval_list), 1),
    }
    return len(train_ids), len(eval_ids)


def get_train_client_ids(experiment_id: int) -> list[str]:
    if experiment_id in _train_client_ids:
        return _train_client_ids[experiment_id]
    return list(get_all_clients(experiment_id).keys())


def get_eval_records(experiment_id: int) -> list[dict]:
    return list(_eval_records.get(experiment_id, []))


def get_eval_meta(experiment_id: int) -> dict:
    return dict(_eval_meta.get(experiment_id, {}))


def get_client(experiment_id: int, client_id: str) -> TelemetryClient | None:
    reg = _registry.get(experiment_id, {})
    return reg.get(client_id)


def get_all_clients(experiment_id: int) -> dict[str, TelemetryClient]:
    return _registry.get(experiment_id, {})


def get_client_records_map(experiment_id: int) -> dict[str, list[dict]]:
    """Training records per client (post holdout split when configured)."""
    if experiment_id in _train_records_map:
        return _train_records_map[experiment_id]
    return {
        cid: client.get_local_records()
        for cid, client in get_all_clients(experiment_id).items()
    }


def get_client_cohort_map(experiment_id: int) -> dict[str, str]:
    return {cid: client.cohort for cid, client in get_all_clients(experiment_id).items()}


def select_clients(
    experiment_id: int, k: int, seed: int, round_number: int
) -> list[str]:
    import numpy as np

    ids = get_train_client_ids(experiment_id)
    if not ids:
        return []
    rng = np.random.default_rng(seed + round_number)
    k = min(k, len(ids))
    return list(rng.choice(ids, size=k, replace=False))


def clear_registry(experiment_id: int | None = None) -> None:
    if experiment_id is None:
        _registry.clear()
        _train_client_ids.clear()
        _train_records_map.clear()
        _eval_records.clear()
        _eval_meta.clear()
    else:
        _registry.pop(experiment_id, None)
        _train_client_ids.pop(experiment_id, None)
        _train_records_map.pop(experiment_id, None)
        _eval_records.pop(experiment_id, None)
        _eval_meta.pop(experiment_id, None)
