"""End-to-end experiment execution."""

import numpy as np
from sqlalchemy.orm import Session

from app.coordinator.client_registry import (
    get_client_records_map,
    populate_registry,
)
from app.coordinator.round_coordinator import RoundCoordinator
from app.learning.centralized_baseline import train_centralized_baseline
from app.learning.model import build_model, weights_to_vector
from app.models import Experiment, RoundMetric
from app.schemas import ExperimentStartRequest


def config_from_experiment(exp: Experiment) -> ExperimentStartRequest:
    """Rebuild a full request from DB row (includes fields not stored on Experiment)."""
    return ExperimentStartRequest(
        name=exp.name,
        mode=exp.mode,  # type: ignore[arg-type]
        dataset=exp.dataset,
        num_clients=exp.num_clients or 500,
        rounds=exp.rounds or 10,
        clients_per_round=exp.clients_per_round or 50,
        local_epochs=exp.local_epochs or 2,
        dropout_rate=exp.dropout_rate or 0.15,
        secure_agg_enabled=bool(exp.secure_agg_enabled),
        dp_enabled=bool(exp.dp_enabled),
        clipping_norm=exp.clipping_norm or 1.0,
        noise_multiplier=exp.noise_multiplier or 0.5,
        epsilon=exp.epsilon or 2.0,
        delta=exp.delta or 1e-6,
    )


class ExperimentRunner:
    def __init__(self, db: Session):
        self.db = db

    def create_experiment(self, config: ExperimentStartRequest) -> Experiment:
        exp = Experiment(
            name=config.name,
            mode=config.mode,
            dataset=config.dataset,
            num_clients=config.num_clients,
            rounds=config.rounds,
            clients_per_round=config.clients_per_round,
            local_epochs=config.local_epochs,
            dp_enabled=config.dp_enabled,
            secure_agg_enabled=config.secure_agg_enabled,
            dropout_rate=config.dropout_rate,
            clipping_norm=config.clipping_norm,
            noise_multiplier=config.noise_multiplier,
            epsilon=config.epsilon,
            delta=config.delta,
            status="created",
        )
        self.db.add(exp)
        self.db.commit()
        self.db.refresh(exp)
        return exp

    def initialize_clients(self, experiment: Experiment, config: ExperimentStartRequest) -> int:
        from app.config_loader import set_active_dataset
        from app.learning.feature_encoding import fit_normalization, reset_normalization
        from app.coordinator.client_registry import (
            configure_train_eval_split,
            get_client_records_map,
            get_train_client_ids,
        )

        set_active_dataset(config.dataset)
        reset_normalization()
        n = populate_registry(
            experiment.id,
            config.num_clients,
            seed=config.random_seed,
            non_iid_severity=config.non_iid_severity,
            dataset=config.dataset,
        )
        configure_train_eval_split(
            experiment.id, seed=config.random_seed, dataset=config.dataset
        )
        # Fit normalization on training clients only (no eval leakage)
        train_records: list = []
        records_map = get_client_records_map(experiment.id)
        for cid in get_train_client_ids(experiment.id):
            train_records.extend(records_map.get(cid, []))
        fit_normalization(train_records)
        return n

    def run_full_experiment(
        self,
        experiment_id: int,
        config: ExperimentStartRequest | None = None,
    ) -> dict:
        exp = self.db.query(Experiment).filter(Experiment.id == experiment_id).first()
        if not exp:
            raise ValueError(f"Experiment {experiment_id} not found")

        if config is None:
            config = config_from_experiment(exp)

        if exp.mode == "centralized_baseline":
            return self._run_centralized(exp, config)

        self.initialize_clients(exp, config)

        import tensorflow as tf

        tf.random.set_seed(config.random_seed)
        model = build_model()
        global_weights = weights_to_vector(model)
        coordinator = RoundCoordinator(self.db, exp.id, config, global_weights)

        exp.status = "running"
        self.db.commit()

        round_results = []
        for r in range(1, (exp.rounds or 10) + 1):
            if exp.mode == "federated_analytics":
                result = coordinator.run_analytics_round(r)
            else:
                result = coordinator.run_learning_round(r)
                global_weights = coordinator.global_weights
            round_results.append(result)

        exp.status = "completed"
        self.db.commit()
        return {"experiment_id": exp.id, "rounds_completed": len(round_results), "results": round_results}

    def run_single_round(self, experiment_id: int, round_number: int | None = None) -> dict:
        exp = self.db.query(Experiment).filter(Experiment.id == experiment_id).first()
        if not exp:
            raise ValueError(f"Experiment {experiment_id} not found")

        config = config_from_experiment(exp)

        from app.coordinator.client_registry import get_all_clients

        if not get_all_clients(exp.id):
            populate_registry(
                exp.id,
                config.num_clients,
                seed=config.random_seed,
                non_iid_severity=config.non_iid_severity,
                dataset=config.dataset,
            )

        existing_rounds = (
            self.db.query(RoundMetric)
            .filter(RoundMetric.experiment_id == experiment_id)
            .count()
        )
        analytics_rounds = (
            self.db.query(RoundMetric)
            .filter(RoundMetric.experiment_id == experiment_id)
            .count()
        )
        from app.models import AnalyticsResult

        ar_count = (
            self.db.query(AnalyticsResult)
            .filter(AnalyticsResult.experiment_id == experiment_id)
            .count()
        )
        next_round = round_number or max(existing_rounds, ar_count // 4, 0) + 1

        import tensorflow as tf

        tf.random.set_seed(config.random_seed)
        coordinator = RoundCoordinator(
            self.db, exp.id, config, weights_to_vector(build_model())
        )

        if exp.mode == "federated_analytics":
            return coordinator.run_analytics_round(next_round)
        return coordinator.run_learning_round(next_round)

    def _run_centralized(self, exp: Experiment, config: ExperimentStartRequest) -> dict:
        from app.config_loader import set_active_dataset
        from app.coordinator.client_registry import (
            configure_train_eval_split,
            get_client_records_map,
            get_eval_records,
            get_train_client_ids,
        )
        from app.learning.feature_encoding import fit_normalization, reset_normalization

        set_active_dataset(config.dataset)
        reset_normalization()
        populate_registry(
            exp.id,
            config.num_clients,
            seed=config.random_seed,
            dataset=config.dataset,
        )
        configure_train_eval_split(
            exp.id, seed=config.random_seed, dataset=config.dataset
        )
        records_map = get_client_records_map(exp.id)
        train_records = []
        for cid in get_train_client_ids(exp.id):
            train_records.extend(records_map.get(cid, []))
        fit_normalization(train_records)

        total_epochs = max(
            (config.local_epochs or 2) * (config.rounds or 1),
            config.local_epochs or 2,
        )
        model, _ = train_centralized_baseline(train_records, epochs=total_epochs)
        from app.learning.evaluator import evaluate_model
        from app.learning.feature_encoding import records_to_xy

        eval_records = get_eval_records(exp.id)
        X_eval, y_eval = records_to_xy(eval_records)
        metrics = evaluate_model(model, X_eval, y_eval)
        self.db.add(
            RoundMetric(
                experiment_id=exp.id,
                round_number=1,
                selected_clients=len(records_map),
                completed_clients=len(records_map),
                dropped_clients=0,
                accuracy=metrics.get("accuracy"),
                roc_auc=metrics.get("roc_auc"),
                precision_score=metrics.get("precision"),
                recall_score=metrics.get("recall"),
                f1_score=metrics.get("f1"),
                train_loss=metrics.get("loss"),
                eval_loss=metrics.get("loss"),
            )
        )
        exp.status = "completed"
        self.db.commit()
        return {"experiment_id": exp.id, "mode": "centralized_baseline", "metrics": metrics}
