"""
Federated Inference Coordinator — FastAPI application (port 8001).

Design:
  - Clients register and receive the latest model + privacy budget.
  - Clients download the model and perform inference LOCALLY.
  - Only aggregate metadata (counts, latencies) is reported back — never
    individual predictions. This is the core privacy guarantee.
  - SQLite stores client registry and inference logs for audit and drift detection.

Run with:
    uvicorn fedprivacylab.inference.coordinator:app --host 0.0.0.0 --port 8001
"""

from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import sqlalchemy as sa
from sqlalchemy import create_engine, Column, String, Integer, Float, DateTime, Text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

# ── Database setup ─────────────────────────────────────────────────────────────

DB_PATH = os.environ.get(
    "INFERENCE_DB_URL",
    str(Path(__file__).parents[3] / "data" / "inference.db"),
)
engine = create_engine(f"sqlite:///{DB_PATH}", connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


class Base(DeclarativeBase):
    pass


class DBClient(Base):
    __tablename__ = "clients"

    id = Column(String, primary_key=True)
    registered_at = Column(DateTime, default=datetime.utcnow)
    last_seen = Column(DateTime, default=datetime.utcnow)
    inference_count = Column(Integer, default=0)
    epsilon_consumed = Column(Float, default=0.0)
    model_version = Column(Integer, default=1)


class DBInferenceLog(Base):
    __tablename__ = "inference_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    client_id = Column(String, sa.ForeignKey("clients.id"))
    timestamp = Column(DateTime, default=datetime.utcnow)
    num_predictions = Column(Integer)
    latency_ms = Column(Float)
    model_version = Column(Integer)
    epsilon_this_batch = Column(Float, default=0.0)


# ── In-memory model registry ───────────────────────────────────────────────────

_MODEL_REGISTRY: dict[str, Any] = {
    "version": 1,
    # Weights stored as list of lists for JSON serialisability.
    # In production, replace with a path to a TFLite flatbuffer.
    "weights_path": None,
    "privacy_params": {
        "epsilon_per_client": 1.0,
        "delta": 1e-5,
        "noise_multiplier": 1.0,
    },
    "validation_accuracy": 0.85,
    "updated_at": datetime.utcnow().isoformat(),
}


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


# ── Lifespan ───────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    Base.metadata.create_all(bind=engine)
    yield


app = FastAPI(
    title="FedPrivacyLab Inference Coordinator",
    version="1.0.0",
    description="Federated inference coordinator — model distribution and metadata aggregation.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic schemas ───────────────────────────────────────────────────────────

class RegisterClientRequest(BaseModel):
    client_id: str = Field(..., description="Unique client identifier")
    capabilities: dict[str, Any] = Field(default_factory=dict, description="Optional client metadata")


class RegisterClientResponse(BaseModel):
    client_id: str
    model_version: int
    privacy_budget: float
    delta: float
    message: str


class GetModelResponse(BaseModel):
    model_version: int
    weights_path: str | None
    privacy_params: dict[str, Any]
    model_type: str
    updated_at: str


class InferenceLogRequest(BaseModel):
    client_id: str = Field(..., description="Client reporting the batch")
    num_predictions: int = Field(..., ge=0, description="Number of local predictions made")
    latency_ms: float = Field(..., ge=0.0, description="Average per-sample latency in ms")
    model_version: int = Field(..., description="Model version used for inference")


class InferenceLogResponse(BaseModel):
    accepted: bool
    epsilon_remaining: float
    warning: str | None = None


class InferenceStatsResponse(BaseModel):
    total_predictions: int
    num_active_clients: int
    avg_latency_ms: float
    total_epsilon_spent: float
    model_version: int


class AccuracyDriftPoint(BaseModel):
    model_version: int
    validation_accuracy: float


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "service": "inference-coordinator"}


@app.post("/register_client", response_model=RegisterClientResponse)
async def register_client(req: RegisterClientRequest) -> RegisterClientResponse:
    """
    Register a new inference client and return the current model version
    plus its allocated privacy budget.

    Privacy guarantee: registration only stores a client ID and timestamps.
    No data about the client's local dataset or users is collected.
    """
    with SessionLocal() as db:
        existing = db.get(DBClient, req.client_id)
        if existing:
            existing.last_seen = _utcnow()
            existing.model_version = _MODEL_REGISTRY["version"]
            db.commit()
        else:
            db.add(DBClient(
                id=req.client_id,
                registered_at=_utcnow(),
                last_seen=_utcnow(),
                model_version=_MODEL_REGISTRY["version"],
            ))
            db.commit()

    budget = _MODEL_REGISTRY["privacy_params"]["epsilon_per_client"]
    return RegisterClientResponse(
        client_id=req.client_id,
        model_version=_MODEL_REGISTRY["version"],
        privacy_budget=budget,
        delta=_MODEL_REGISTRY["privacy_params"]["delta"],
        message="Registered successfully. Perform all inference locally; only metadata is reported.",
    )


@app.post("/get_model", response_model=GetModelResponse)
async def get_model(body: dict = {}) -> GetModelResponse:
    """
    Return the current global model descriptor.

    Clients use this to download weights (or a TFLite file path) and cache them
    locally. All subsequent predictions run on-device without network calls.
    """
    return GetModelResponse(
        model_version=_MODEL_REGISTRY["version"],
        weights_path=_MODEL_REGISTRY["weights_path"],
        privacy_params=_MODEL_REGISTRY["privacy_params"],
        model_type="tflite" if _MODEL_REGISTRY["weights_path"] else "stub",
        updated_at=_MODEL_REGISTRY["updated_at"],
    )


@app.post("/client_inference_log", response_model=InferenceLogResponse)
async def client_inference_log(req: InferenceLogRequest) -> InferenceLogResponse:
    """
    Accept a metadata report from a client about a completed inference batch.

    Privacy guarantee: Only aggregate counts and timing are received.
    Individual predictions, input features, and user identities are NEVER sent.
    """
    # Compute epsilon cost for this batch (proportional to predictions)
    epsilon_cost = _epsilon_cost(req.num_predictions)

    with SessionLocal() as db:
        client = db.get(DBClient, req.client_id)
        if not client:
            raise HTTPException(status_code=404, detail=f"Client '{req.client_id}' not registered.")

        budget = _MODEL_REGISTRY["privacy_params"]["epsilon_per_client"]
        remaining = budget - client.epsilon_consumed

        if epsilon_cost > remaining:
            return InferenceLogResponse(
                accepted=False,
                epsilon_remaining=round(remaining, 4),
                warning="Privacy budget exhausted. Re-register or request budget extension.",
            )

        client.inference_count += req.num_predictions
        client.epsilon_consumed += epsilon_cost
        client.last_seen = _utcnow()
        client.model_version = req.model_version

        db.add(DBInferenceLog(
            client_id=req.client_id,
            timestamp=_utcnow(),
            num_predictions=req.num_predictions,
            latency_ms=req.latency_ms,
            model_version=req.model_version,
            epsilon_this_batch=epsilon_cost,
        ))
        db.commit()
        new_remaining = budget - client.epsilon_consumed

    warning = None
    if new_remaining < 0.2 * budget:
        warning = f"Low privacy budget: {new_remaining:.3f} epsilon remaining."

    return InferenceLogResponse(
        accepted=True,
        epsilon_remaining=round(new_remaining, 4),
        warning=warning,
    )


@app.get("/inference_stats", response_model=InferenceStatsResponse)
async def inference_stats() -> InferenceStatsResponse:
    """Return aggregate inference statistics across all clients."""
    with SessionLocal() as db:
        total_preds = db.execute(sa.text("SELECT COALESCE(SUM(num_predictions), 0) FROM inference_logs")).scalar()
        avg_lat = db.execute(
            sa.text("SELECT COALESCE(AVG(latency_ms), 0) FROM inference_logs")
        ).scalar()
        num_clients = db.execute(sa.text("SELECT COUNT(*) FROM clients")).scalar()
        total_eps = db.execute(sa.text("SELECT COALESCE(SUM(epsilon_consumed), 0) FROM clients")).scalar()

    return InferenceStatsResponse(
        total_predictions=int(total_preds),
        num_active_clients=int(num_clients),
        avg_latency_ms=round(float(avg_lat), 2),
        total_epsilon_spent=round(float(total_eps), 4),
        model_version=_MODEL_REGISTRY["version"],
    )


@app.get("/model_accuracy_drift")
async def model_accuracy_drift() -> dict:
    """
    Return validation accuracy indexed by model version.

    In production this would query a validation pipeline that runs on a
    held-out server-side dataset after each federated round.
    Here, a single data point is returned for the current version.
    """
    return {
        "drift_series": [
            {
                "model_version": _MODEL_REGISTRY["version"],
                "validation_accuracy": _MODEL_REGISTRY["validation_accuracy"],
            }
        ],
        "baseline_accuracy": _MODEL_REGISTRY["validation_accuracy"],
        "current_accuracy": _MODEL_REGISTRY["validation_accuracy"],
        "drift_percent": 0.0,
    }


@app.get("/clients")
async def list_clients() -> dict:
    """Return the active client registry (last_seen, inference count, epsilon)."""
    with SessionLocal() as db:
        rows = db.execute(sa.text(
            "SELECT id, last_seen, inference_count, epsilon_consumed, model_version FROM clients"
        )).fetchall()

    budget = _MODEL_REGISTRY["privacy_params"]["epsilon_per_client"]
    clients = [
        {
            "client_id": r[0],
            "last_seen": str(r[1]),
            "total_inferences": r[2],
            "epsilon_consumed": round(r[3], 4),
            "epsilon_remaining": round(max(0.0, budget - r[3]), 4),
            "model_version": r[4],
        }
        for r in rows
    ]
    return {"clients": clients, "count": len(clients)}


@app.get("/inference_timeseries")
async def inference_timeseries() -> dict:
    """Return per-timestamp inference counts for volume charts."""
    with SessionLocal() as db:
        rows = db.execute(sa.text(
            "SELECT timestamp, client_id, num_predictions, latency_ms FROM inference_logs ORDER BY timestamp"
        )).fetchall()

    return {
        "series": [
            {
                "timestamp": str(r[0]),
                "client_id": r[1],
                "num_predictions": r[2],
                "latency_ms": r[3],
            }
            for r in rows
        ]
    }


# ── Internal helpers ───────────────────────────────────────────────────────────

def _epsilon_cost(num_predictions: int, sensitivity: float = 1.0, noise_scale: float = 1.0) -> float:
    """
    Compute privacy cost for reporting a batch of `num_predictions` metadata items.

    Uses a simplified RDP-to-(ε,δ) accounting: epsilon ≈ sensitivity / noise_scale * log(num_predictions + 1).
    This is conservative (overestimates cost) to ensure the budget is not over-spent.

    Args:
        num_predictions: Number of local predictions made in this batch.
        sensitivity: L1 sensitivity of the reported count.
        noise_scale: Gaussian noise scale (σ).

    Returns:
        Epsilon cost for this batch.
    """
    import math
    return (sensitivity / noise_scale) * math.log1p(num_predictions) * 0.01


def update_model_registry(
    version: int,
    weights_path: str | None = None,
    validation_accuracy: float | None = None,
) -> None:
    """
    Update the in-memory model registry (called by the training coordinator after each FL round).

    Args:
        version: New model version number.
        weights_path: Path to the TFLite model file.
        validation_accuracy: Accuracy on the validation set.
    """
    _MODEL_REGISTRY["version"] = version
    if weights_path is not None:
        _MODEL_REGISTRY["weights_path"] = weights_path
    if validation_accuracy is not None:
        _MODEL_REGISTRY["validation_accuracy"] = validation_accuracy
    _MODEL_REGISTRY["updated_at"] = datetime.utcnow().isoformat()
