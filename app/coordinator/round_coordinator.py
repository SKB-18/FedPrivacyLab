"""Per-round orchestration for analytics and learning."""

import hashlib
import json
from typing import Any

import numpy as np
from sqlalchemy.orm import Session

from app.analytics.federated_analytics import run_full_analytics_suite
from app.coordinator.client_registry import get_client_cohort_map
from app.clients.local_training import train_local_model
from app.coordinator.client_registry import get_client_records_map, select_clients
from app.learning.evaluator import evaluate_model
from app.learning.fedavg import aggregate_client_deltas
from app.learning.model import build_model, records_to_xy, set_model_weights, weights_to_vector
from app.models import AnalyticsResult, ClientParticipation, PrivacyReport, RoundMetric
from app.privacy.dp_mechanisms import add_gaussian_noise_to_update
from app.privacy.privacy_accountant import PrivacyAccountant
from app.privacy.risk_report import build_privacy_report
from app.privacy.secure_aggregation import mask_update, aggregate_masked_updates
from app.privacy.update_clipping import clip_update
from app.schemas import ExperimentStartRequest
from app.simulation.dropout_simulator import apply_dropout


class RoundCoordinator:
    def __init__(
        self,
        db: Session,
        experiment_id: int,
        config: ExperimentStartRequest | dict,
        global_weights: np.ndarray | None = None,
    ):
        self.db = db
        self.experiment_id = experiment_id
        if isinstance(config, dict):
            self.config = ExperimentStartRequest(**config)
        else:
            self.config = config
        self.global_weights = global_weights
        self.current_round = 0
        self._accountant = PrivacyAccountant(
            self.config.epsilon, self.config.delta
        )

    def _cfg(self, key: str, default=None):
        return getattr(self.config, key, default)

    def select_and_apply_dropout(self, round_number: int) -> tuple[list[str], list[str]]:
        selected = select_clients(
            self.experiment_id,
            self.config.clients_per_round,
            self.config.random_seed,
            round_number,
        )
        completed, dropped = apply_dropout(
            selected,
            self.config.dropout_rate,
            seed=self.config.random_seed + round_number,
        )
        return selected, completed, dropped

    def run_analytics_round(self, round_number: int) -> dict[str, Any]:
        selected, completed, dropped = self.select_and_apply_dropout(round_number)
        records_map = get_client_records_map(self.experiment_id)

        cohort_map = get_client_cohort_map(self.experiment_id)
        results, meta = run_full_analytics_suite(
            records_map,
            completed,
            cohort_map=cohort_map,
            dp_enabled=self.config.dp_enabled,
            epsilon=self.config.epsilon,
            round_number=round_number,
        )

        for r in results:
            self.db.add(
                AnalyticsResult(
                    experiment_id=self.experiment_id,
                    round_number=round_number,
                    metric_name=r.metric_name,
                    feature=r.feature,
                    true_value=r.true_value,
                    federated_value=r.federated_value,
                    dp_noisy_value=r.dp_noisy_value,
                    epsilon=r.epsilon,
                    absolute_error=r.absolute_error,
                    relative_error=r.relative_error,
                    suppressed=r.suppressed,
                )
            )

        self._save_participation(round_number, selected, completed, dropped)
        self._save_privacy_report(round_number)
        self.db.commit()

        return {
            "round": round_number,
            "metrics": [r.__dict__ for r in results],
            "meta": meta,
            "selected": len(selected),
            "completed": len(completed),
            "dropped": len(dropped),
        }

    def run_learning_round(self, round_number: int) -> dict[str, Any]:
        if self.global_weights is None:
            model = build_model()
            self.global_weights = weights_to_vector(model)

        selected, completed, dropped = self.select_and_apply_dropout(round_number)
        records_map = get_client_records_map(self.experiment_id)

        deltas = []
        sizes = []
        norms = []
        clipped_flags = []
        masked_updates = []
        masks = []

        secure = self.config.secure_agg_enabled or self.config.mode in (
            "fedavg_secureagg",
            "fedavg_secureagg_dp",
        )
        use_dp = self.config.dp_enabled or self.config.mode == "fedavg_secureagg_dp"

        for i, cid in enumerate(completed):
            records = records_map.get(cid, [])
            delta, norm, n = train_local_model(
                records,
                self.global_weights,
                local_epochs=self.config.local_epochs,
                seed=self.config.random_seed + round_number + i,
            )
            if n == 0:
                continue

            clipped_delta, was_clipped = clip_update(delta, self.config.clipping_norm)
            deltas.append(clipped_delta)
            sizes.append(n)
            norms.append(norm)
            clipped_flags.append(was_clipped)

            self.db.add(
                ClientParticipation(
                    experiment_id=self.experiment_id,
                    round_number=round_number,
                    client_id_hash=hashlib.sha256(cid.encode()).hexdigest()[:16],
                    selected=True,
                    completed=True,
                    num_examples=n,
                    update_norm=norm,
                    clipped=was_clipped,
                )
            )

        for cid in dropped:
            self.db.add(
                ClientParticipation(
                    experiment_id=self.experiment_id,
                    round_number=round_number,
                    client_id_hash=hashlib.sha256(cid.encode()).hexdigest()[:16],
                    selected=True,
                    completed=False,
                )
            )

        if not deltas:
            return {"round": round_number, "status": "no_updates"}

        # FedAvg weighted mean of clipped deltas (secure masks are simulated only).
        agg_delta = aggregate_client_deltas(deltas, sizes)
        if secure:
            masked_updates, masks = [], []
            for i, clipped_delta in enumerate(deltas):
                m_upd, mask = mask_update(
                    clipped_delta,
                    self.config.random_seed + round_number * 1000 + i,
                )
                masked_updates.append(m_upd)
                masks.append(mask)
            _ = aggregate_masked_updates(masked_updates, masks)

        if use_dp:
            n_part = max(len(completed), 1)
            effective_noise = self.config.noise_multiplier / np.sqrt(n_part)
            rng = np.random.default_rng(
                self.config.random_seed + round_number * 10_000
            )
            agg_delta = add_gaussian_noise_to_update(
                agg_delta,
                self.config.clipping_norm,
                effective_noise,
                rng=rng,
            )
            self._accountant.charge(self.config.epsilon / max(self.config.rounds, 1))

        self.global_weights = self.global_weights + agg_delta

        from app.coordinator.client_registry import get_eval_records

        eval_records = get_eval_records(self.experiment_id)
        if not eval_records:
            for cid in completed:
                eval_records.extend(records_map.get(cid, []))
        X, y = records_to_xy(eval_records)
        eval_model = build_model()
        set_model_weights(eval_model, self.global_weights)
        metrics = evaluate_model(eval_model, X, y)

        train_loss = metrics.get("loss")
        self.db.add(
            RoundMetric(
                experiment_id=self.experiment_id,
                round_number=round_number,
                selected_clients=len(selected),
                completed_clients=len(completed),
                dropped_clients=len(dropped),
                train_loss=train_loss,
                eval_loss=train_loss,
                accuracy=metrics.get("accuracy"),
                roc_auc=metrics.get("roc_auc"),
                precision_score=metrics.get("precision"),
                recall_score=metrics.get("recall"),
                f1_score=metrics.get("f1"),
            )
        )
        self._save_privacy_report(round_number)
        self.db.commit()

        return {
            "round": round_number,
            "metrics": metrics,
            "selected": len(selected),
            "completed": len(completed),
            "dropped": len(dropped),
            "global_weights_norm": float(np.linalg.norm(self.global_weights)),
        }

    def _save_participation(
        self,
        round_number: int,
        selected: list[str],
        completed: list[str],
        dropped: list[str],
    ) -> None:
        completed_set = set(completed)
        for cid in selected:
            self.db.add(
                ClientParticipation(
                    experiment_id=self.experiment_id,
                    round_number=round_number,
                    client_id_hash=hashlib.sha256(cid.encode()).hexdigest()[:16],
                    selected=True,
                    completed=cid in completed_set,
                )
            )

    def _save_privacy_report(self, round_number: int) -> None:
        report = build_privacy_report(self.experiment_id, self.config)
        self.db.add(
            PrivacyReport(
                experiment_id=self.experiment_id,
                round_number=round_number,
                privacy_mode=report.privacy_mode,
                clipping_norm=report.clipping_norm,
                noise_multiplier=report.noise_multiplier,
                epsilon=report.epsilon,
                delta=report.delta,
                secure_agg_enabled=report.secure_agg_enabled,
                risk_level=report.risk_level,
                notes=json.dumps(report.notes),
            )
        )
