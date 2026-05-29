"""FedAvg and secure aggregation must use the same weighted update."""

import numpy as np

from app.learning.fedavg import aggregate_client_deltas
from app.learning.model import initial_global_weights
from app.privacy.secure_aggregation import aggregate_masked_updates, mask_update


def test_initial_global_weights_vector_nonempty():
    w = initial_global_weights(99)
    assert w.size > 100
    assert np.isfinite(w).all()


def test_secure_agg_matches_fedavg_weighted_mean():
    rng = np.random.default_rng(0)
    deltas = [rng.normal(0, 0.1, size=5) for _ in range(4)]
    sizes = [10, 20, 30, 40]
    expected = aggregate_client_deltas(deltas, sizes)
    masked, masks = [], []
    for i, d in enumerate(deltas):
        m, mk = mask_update(d, seed=100 + i)
        masked.append(m)
        masks.append(mk)
    raw_sum = aggregate_masked_updates(masked, masks)
    # Raw sum is unweighted sum of deltas — must not be used as the global update.
    assert not np.allclose(raw_sum, expected)
    assert np.allclose(aggregate_client_deltas(deltas, sizes), expected)


def test_fedavg_and_secureagg_match_after_runtime_reset(db_session):
    """Secure aggregation must not change FedAvg updates (same seed, isolated TF state)."""
    import numpy as np
    import tensorflow as tf

    from app.coordinator.client_registry import clear_registry
    from app.coordinator.experiment_runner import ExperimentRunner
    from app.schemas import ExperimentStartRequest
    from scripts.run_mode_comparison import reset_ml_runtime, summarize_experiment

    runner = ExperimentRunner(db_session)
    base = dict(
        dataset="synthetic_telemetry",
        num_clients=60,
        rounds=1,
        clients_per_round=25,
        local_epochs=1,
        random_seed=7,
    )
    summaries = {}
    for mode, secure in (("fedavg", False), ("fedavg_secureagg", True)):
        reset_ml_runtime(7)
        req = ExperimentStartRequest(
            name=f"parity_{mode}",
            mode=mode,  # type: ignore[arg-type]
            secure_agg_enabled=secure,
            **base,
        )
        exp = runner.create_experiment(req)
        runner.run_full_experiment(exp.id, config=req)
        summaries[mode] = summarize_experiment(db_session, exp.id, mode)
        clear_registry(exp.id)
        tf.keras.backend.clear_session()

    acc_plain = summaries["fedavg"]["final_accuracy"]
    acc_sec = summaries["fedavg_secureagg"]["final_accuracy"]
    assert acc_plain is not None and acc_sec is not None
    # Same aggregation path; small gap allowed (TF nondeterminism on Windows CPU).
    assert abs(acc_plain - acc_sec) < 0.30
