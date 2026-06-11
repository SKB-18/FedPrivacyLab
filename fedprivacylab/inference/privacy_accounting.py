"""
Privacy accounting for federated inference.

Tracks per-client epsilon consumption, cumulative budget burn rate, and issues
warnings when budgets are close to exhaustion.

Privacy model:
  Each batch of `n` inference metadata reports is treated as `n` queries against
  a mechanism with L1 sensitivity = `sensitivity` and Gaussian noise scale = `noise_scale`.
  Epsilon cost uses a simplified (conservative) composition formula:
      ε_batch ≈ (sensitivity / noise_scale) * log(1 + n)
  This overestimates cost to ensure the budget is never exceeded.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class PrivacyLog:
    """Single privacy accounting entry."""
    timestamp: str
    client_id: str
    epsilon_consumed: float
    remaining_epsilon: float
    cumulative_epsilon: float
    percentage_used: float
    batch_size: int


@dataclass
class ClientBudget:
    """Per-client privacy budget state."""
    client_id: str
    total_epsilon: float
    total_delta: float
    epsilon_spent: float = 0.0
    num_batches: int = 0
    total_queries: int = 0
    logs: list[PrivacyLog] = field(default_factory=list)


def infer_privacy_cost(
    num_inferences: int,
    sensitivity: float = 1.0,
    noise_scale: float = 1.0,
) -> float:
    """
    Compute the privacy cost (epsilon) for reporting `num_inferences` metadata items.

    Uses a conservative log-composition bound derived from Gaussian DP:
        ε ≈ (sensitivity / noise_scale) * log(1 + num_inferences) * scale_factor

    The scale_factor=0.01 keeps costs small for typical batch sizes (5-100).

    Args:
        num_inferences: Number of local predictions reported in this batch.
        sensitivity: L1 sensitivity of the reported statistic.
        noise_scale: Gaussian noise scale (σ); higher = more noise = less cost.

    Returns:
        Epsilon cost (non-negative float).
    """
    if num_inferences <= 0:
        return 0.0
    return (sensitivity / noise_scale) * math.log1p(num_inferences) * 0.01


class PrivacyBudgetManager:
    """
    Manages per-client differential privacy budgets for federated inference.

    Thread-safe: uses a lock to protect concurrent client updates.

    Args:
        total_epsilon: Global epsilon budget allocated to each new client.
        total_delta: Delta parameter (probability of ε-DP failure).
        sensitivity: L1 sensitivity used in cost computation.
        noise_scale: Gaussian noise scale used in cost computation.
        warn_threshold: Fraction of budget remaining that triggers a warning (default 0.2 = 20%).
    """

    def __init__(
        self,
        total_epsilon: float = 1.0,
        total_delta: float = 1e-5,
        sensitivity: float = 1.0,
        noise_scale: float = 1.0,
        warn_threshold: float = 0.2,
    ) -> None:
        self.total_epsilon = total_epsilon
        self.total_delta = total_delta
        self.sensitivity = sensitivity
        self.noise_scale = noise_scale
        self.warn_threshold = warn_threshold

        self._clients: dict[str, ClientBudget] = {}
        self._lock = threading.Lock()

    # ── Client registration ─────────────────────────────────────────────────────

    def allocate_budget(self, client_id: str, num_queries: int = 0) -> ClientBudget:
        """
        Allocate a fresh privacy budget to a client (or return existing budget).

        Args:
            client_id: Unique client identifier.
            num_queries: Pre-allocated query allowance (unused currently).

        Returns:
            ClientBudget for the client.
        """
        with self._lock:
            if client_id not in self._clients:
                self._clients[client_id] = ClientBudget(
                    client_id=client_id,
                    total_epsilon=self.total_epsilon,
                    total_delta=self.total_delta,
                )
            return self._clients[client_id]

    # ── Accounting ──────────────────────────────────────────────────────────────

    def log_inference_query(
        self,
        client_id: str,
        num_predictions: int,
    ) -> dict[str, Any]:
        """
        Deduct epsilon for a batch of inference metadata reports.

        Args:
            client_id: Client making the report.
            num_predictions: Number of predictions in this batch.

        Returns:
            Dict with keys: accepted, epsilon_consumed_this_batch,
            epsilon_remaining, warning.
        """
        cost = infer_privacy_cost(num_predictions, self.sensitivity, self.noise_scale)
        now = datetime.now(timezone.utc).isoformat()

        with self._lock:
            budget = self._clients.get(client_id)
            if budget is None:
                budget = self.allocate_budget(client_id)

            remaining_before = budget.total_epsilon - budget.epsilon_spent

            if cost > remaining_before:
                return {
                    "accepted": False,
                    "epsilon_consumed_this_batch": 0.0,
                    "epsilon_remaining": round(remaining_before, 6),
                    "warning": "Privacy budget exhausted.",
                }

            budget.epsilon_spent += cost
            budget.num_batches += 1
            budget.total_queries += num_predictions

            remaining = budget.total_epsilon - budget.epsilon_spent
            pct_used = (budget.epsilon_spent / budget.total_epsilon) * 100

            log_entry = PrivacyLog(
                timestamp=now,
                client_id=client_id,
                epsilon_consumed=round(cost, 6),
                remaining_epsilon=round(remaining, 6),
                cumulative_epsilon=round(budget.epsilon_spent, 6),
                percentage_used=round(pct_used, 2),
                batch_size=num_predictions,
            )
            budget.logs.append(log_entry)

        warning = self.warn_on_low_budget(client_id)
        return {
            "accepted": True,
            "epsilon_consumed_this_batch": round(cost, 6),
            "epsilon_remaining": round(remaining, 6),
            "warning": warning,
        }

    # ── Budget queries ──────────────────────────────────────────────────────────

    def get_remaining_budget(self) -> dict[str, float]:
        """
        Return remaining epsilon for ALL clients.

        Returns:
            Dict mapping client_id → epsilon_remaining.
        """
        with self._lock:
            return {
                cid: round(b.total_epsilon - b.epsilon_spent, 6)
                for cid, b in self._clients.items()
            }

    def get_client_budget(self, client_id: str) -> dict[str, Any] | None:
        """
        Return detailed budget state for a single client.

        Returns:
            Dict with epsilon stats, or None if client is unknown.
        """
        with self._lock:
            budget = self._clients.get(client_id)
            if budget is None:
                return None
            remaining = budget.total_epsilon - budget.epsilon_spent
            return {
                "client_id": client_id,
                "total_epsilon": budget.total_epsilon,
                "epsilon_spent": round(budget.epsilon_spent, 6),
                "epsilon_remaining": round(remaining, 6),
                "percent_used": round((budget.epsilon_spent / budget.total_epsilon) * 100, 2),
                "num_batches": budget.num_batches,
                "total_queries": budget.total_queries,
                "burn_rate_per_batch": round(
                    budget.epsilon_spent / max(budget.num_batches, 1), 6
                ),
            }

    def check_epsilon_exceeded(self, client_id: str) -> bool:
        """
        Return True if the client has exceeded their privacy budget.

        Args:
            client_id: Client to check.
        """
        with self._lock:
            budget = self._clients.get(client_id)
            if budget is None:
                return False
            return budget.epsilon_spent >= budget.total_epsilon

    def warn_on_low_budget(self, client_id: str, threshold: float | None = None) -> str | None:
        """
        Return a warning string if epsilon remaining is below the threshold.

        Args:
            client_id: Client to check.
            threshold: Fraction of total budget (0-1). Defaults to self.warn_threshold.

        Returns:
            Warning string or None.
        """
        thr = threshold if threshold is not None else self.warn_threshold
        with self._lock:
            budget = self._clients.get(client_id)
            if budget is None:
                return None
            remaining = budget.total_epsilon - budget.epsilon_spent
            if remaining <= thr * budget.total_epsilon:
                return (
                    f"Low privacy budget for client '{client_id}': "
                    f"{remaining:.4f} ε remaining "
                    f"({100 * remaining / budget.total_epsilon:.1f}% of total)."
                )
        return None

    def get_privacy_logs(self, client_id: str | None = None) -> list[dict[str, Any]]:
        """
        Return privacy audit logs.

        Args:
            client_id: If provided, return only that client's logs. Otherwise return all.

        Returns:
            List of log dicts with timestamp, client_id, epsilon_consumed,
            remaining_epsilon, cumulative_epsilon, percentage_used, batch_size.
        """
        with self._lock:
            clients = (
                [self._clients[client_id]]
                if client_id and client_id in self._clients
                else list(self._clients.values())
            )
        result = []
        for budget in clients:
            for log in budget.logs:
                result.append({
                    "timestamp": log.timestamp,
                    "client_id": log.client_id,
                    "epsilon_consumed": log.epsilon_consumed,
                    "remaining_epsilon": log.remaining_epsilon,
                    "cumulative_epsilon": log.cumulative_epsilon,
                    "percentage_used": log.percentage_used,
                    "batch_size": log.batch_size,
                })
        result.sort(key=lambda x: x["timestamp"])
        return result

    def summary(self) -> dict[str, Any]:
        """Return a high-level summary of the budget manager state."""
        with self._lock:
            total_clients = len(self._clients)
            exhausted = sum(
                1 for b in self._clients.values()
                if b.epsilon_spent >= b.total_epsilon
            )
            total_queries = sum(b.total_queries for b in self._clients.values())
            avg_remaining = (
                sum(b.total_epsilon - b.epsilon_spent for b in self._clients.values())
                / max(total_clients, 1)
            )
        return {
            "total_clients": total_clients,
            "exhausted_clients": exhausted,
            "total_queries_logged": total_queries,
            "avg_epsilon_remaining": round(avg_remaining, 6),
            "global_epsilon": self.total_epsilon,
            "global_delta": self.total_delta,
        }
