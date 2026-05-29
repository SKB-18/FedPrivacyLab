"""Simplified privacy budget accounting."""

from dataclasses import dataclass


@dataclass
class PrivacyBudget:
    epsilon_spent: float
    delta: float
    rounds: int
    mechanism: str


class PrivacyAccountant:
    """Tracks cumulative epsilon across rounds (simplified composition)."""

    def __init__(self, target_epsilon: float, target_delta: float):
        self.target_epsilon = target_epsilon
        self.target_delta = target_delta
        self.spent_epsilon = 0.0
        self.rounds = 0

    def charge(self, epsilon_per_round: float) -> PrivacyBudget:
        self.spent_epsilon += epsilon_per_round
        self.rounds += 1
        return PrivacyBudget(
            epsilon_spent=self.spent_epsilon,
            delta=self.target_delta,
            rounds=self.rounds,
            mechanism="gaussian_simplified",
        )

    @property
    def remaining_epsilon(self) -> float:
        return max(0.0, self.target_epsilon - self.spent_epsilon)

    @property
    def budget_exceeded(self) -> bool:
        return self.spent_epsilon > self.target_epsilon
