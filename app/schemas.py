"""Pydantic request/response schemas."""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

ExperimentMode = Literal[
    "centralized_baseline",
    "federated_analytics",
    "fedavg",
    "fedavg_secureagg",
    "fedavg_secureagg_dp",
]


class ExperimentStartRequest(BaseModel):
    name: str
    mode: ExperimentMode
    dataset: str = "synthetic_telemetry"
    num_clients: int = Field(default=500, ge=5, le=50000)
    rounds: int = Field(default=10, ge=1, le=500)
    clients_per_round: int = Field(default=50, ge=1)
    local_epochs: int = Field(default=2, ge=1, le=20)
    dropout_rate: float = Field(default=0.15, ge=0.0, le=0.9)
    secure_agg_enabled: bool = False
    dp_enabled: bool = False
    clipping_norm: float = Field(default=1.0, gt=0)
    noise_multiplier: float = Field(default=0.5, ge=0)
    epsilon: float = Field(default=2.0, gt=0)
    delta: float = Field(default=1e-6, gt=0, lt=1)
    non_iid_severity: float = Field(default=1.0, ge=0, le=2.0)
    random_seed: int = 42


class ExperimentResponse(BaseModel):
    id: int
    name: str
    mode: str
    dataset: str
    num_clients: int | None
    rounds: int | None
    status: str
    created_at: datetime

    model_config = {"from_attributes": True}


class RoundMetricResponse(BaseModel):
    round_number: int
    selected_clients: int | None
    completed_clients: int | None
    dropped_clients: int | None
    train_loss: float | None
    eval_loss: float | None
    accuracy: float | None
    roc_auc: float | None
    precision_score: float | None
    recall_score: float | None
    f1_score: float | None

    model_config = {"from_attributes": True}


class AnalyticsResultResponse(BaseModel):
    round_number: int
    metric_name: str
    feature: str | None
    true_value: float | None
    federated_value: float | None
    dp_noisy_value: float | None
    epsilon: float | None
    absolute_error: float | None
    relative_error: float | None
    suppressed: bool

    model_config = {"from_attributes": True}


class PrivacyReportResponse(BaseModel):
    experiment_id: int
    privacy_mode: str
    raw_data_centralized: bool
    individual_updates_visible_to_server: bool
    client_updates_clipped: bool
    dp_noise_added: bool
    secure_agg_enabled: bool
    risk_level: str
    clipping_norm: float | None = None
    noise_multiplier: float | None = None
    epsilon: float | None = None
    delta: float | None = None
    notes: list[str]


class ClientParticipationResponse(BaseModel):
    round_number: int
    client_id_hash: str
    selected: bool
    completed: bool
    num_examples: int | None
    update_norm: float | None
    clipped: bool

    model_config = {"from_attributes": True}


class SimulateClientsRequest(BaseModel):
    num_clients: int = Field(default=500, ge=5, le=50000)
    min_records: int = Field(default=5, ge=1)
    max_records: int = Field(default=50, ge=1)
    random_seed: int = 42
    non_iid_severity: float = Field(default=1.0, ge=0, le=2.0)


class SimulateNonIIDRequest(BaseModel):
    num_clients: int = 500
    severity: float = Field(default=1.0, ge=0, le=2.0)
    random_seed: int = 42


class HealthResponse(BaseModel):
    status: str
    version: str


class RunRoundResponse(BaseModel):
    experiment_id: int
    round_number: int
    status: str
    metrics: dict[str, Any] | list[Any] | None = None
