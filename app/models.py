"""SQLAlchemy ORM models matching architecture spec schema."""

from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class Experiment(Base):
    __tablename__ = "experiments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    mode: Mapped[str] = mapped_column(String, nullable=False)
    dataset: Mapped[str] = mapped_column(String, nullable=False)
    num_clients: Mapped[int | None] = mapped_column(Integer)
    rounds: Mapped[int | None] = mapped_column(Integer)
    clients_per_round: Mapped[int | None] = mapped_column(Integer)
    local_epochs: Mapped[int | None] = mapped_column(Integer)
    dp_enabled: Mapped[bool] = mapped_column(Boolean, default=False)
    secure_agg_enabled: Mapped[bool] = mapped_column(Boolean, default=False)
    dropout_rate: Mapped[float | None] = mapped_column(Float)
    clipping_norm: Mapped[float | None] = mapped_column(Float)
    noise_multiplier: Mapped[float | None] = mapped_column(Float)
    epsilon: Mapped[float | None] = mapped_column(Float)
    delta: Mapped[float | None] = mapped_column(Float)
    status: Mapped[str] = mapped_column(String, default="created")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class RoundMetric(Base):
    __tablename__ = "round_metrics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    experiment_id: Mapped[int] = mapped_column(Integer, nullable=False)
    round_number: Mapped[int] = mapped_column(Integer, nullable=False)
    selected_clients: Mapped[int | None] = mapped_column(Integer)
    completed_clients: Mapped[int | None] = mapped_column(Integer)
    dropped_clients: Mapped[int | None] = mapped_column(Integer)
    train_loss: Mapped[float | None] = mapped_column(Float)
    eval_loss: Mapped[float | None] = mapped_column(Float)
    accuracy: Mapped[float | None] = mapped_column(Float)
    roc_auc: Mapped[float | None] = mapped_column(Float)
    precision_score: Mapped[float | None] = mapped_column(Float)
    recall_score: Mapped[float | None] = mapped_column(Float)
    f1_score: Mapped[float | None] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class AnalyticsResult(Base):
    __tablename__ = "analytics_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    experiment_id: Mapped[int] = mapped_column(Integer, nullable=False)
    round_number: Mapped[int] = mapped_column(Integer, nullable=False)
    metric_name: Mapped[str] = mapped_column(String, nullable=False)
    feature: Mapped[str | None] = mapped_column(String)
    true_value: Mapped[float | None] = mapped_column(Float)
    federated_value: Mapped[float | None] = mapped_column(Float)
    dp_noisy_value: Mapped[float | None] = mapped_column(Float)
    epsilon: Mapped[float | None] = mapped_column(Float)
    absolute_error: Mapped[float | None] = mapped_column(Float)
    relative_error: Mapped[float | None] = mapped_column(Float)
    suppressed: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class PrivacyReport(Base):
    __tablename__ = "privacy_reports"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    experiment_id: Mapped[int] = mapped_column(Integer, nullable=False)
    round_number: Mapped[int | None] = mapped_column(Integer)
    privacy_mode: Mapped[str | None] = mapped_column(String)
    clipping_norm: Mapped[float | None] = mapped_column(Float)
    noise_multiplier: Mapped[float | None] = mapped_column(Float)
    epsilon: Mapped[float | None] = mapped_column(Float)
    delta: Mapped[float | None] = mapped_column(Float)
    secure_agg_enabled: Mapped[bool | None] = mapped_column(Boolean)
    contribution_bound: Mapped[int | None] = mapped_column(Integer)
    risk_level: Mapped[str | None] = mapped_column(String)
    notes: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class ClientParticipation(Base):
    __tablename__ = "client_participation"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    experiment_id: Mapped[int] = mapped_column(Integer, nullable=False)
    round_number: Mapped[int] = mapped_column(Integer, nullable=False)
    client_id_hash: Mapped[str] = mapped_column(String, nullable=False)
    selected: Mapped[bool] = mapped_column(Boolean, default=False)
    completed: Mapped[bool] = mapped_column(Boolean, default=False)
    num_examples: Mapped[int | None] = mapped_column(Integer)
    update_norm: Mapped[float | None] = mapped_column(Float)
    clipped: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
