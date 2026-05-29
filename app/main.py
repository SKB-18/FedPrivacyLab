"""FedPrivacyLab FastAPI application entry point."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import __version__
from app.api.analytics_routes import router as analytics_router
from app.api.experiment_routes import router as experiment_router
from app.api.learning_routes import router as learning_router
from app.api.metrics_routes import router as metrics_router
from app.api.privacy_routes import router as privacy_router
from app.coordinator.client_registry import populate_registry
from app.database import init_db
from app.schemas import HealthResponse, SimulateClientsRequest, SimulateNonIIDRequest
from app.simulation.non_iid_partition import assign_client_profiles


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(
    title="FedPrivacyLab",
    description="Federated Analytics and Privacy-Preserving ML Workbench",
    version=__version__,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(experiment_router)
app.include_router(metrics_router)
app.include_router(analytics_router)
app.include_router(privacy_router)
app.include_router(learning_router)


@app.get("/api/v1/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="healthy", version=__version__)


@app.post("/api/v1/simulate/clients")
def simulate_clients(request: SimulateClientsRequest):
    """Preview client generation without starting an experiment."""
    from app.simulation.synthetic_telemetry import generate_synthetic_clients

    clients = generate_synthetic_clients(
        num_clients=min(request.num_clients, 1000),
        min_records=request.min_records,
        max_records=request.max_records,
        seed=request.random_seed,
        non_iid_severity=request.non_iid_severity,
    )
    profile_counts: dict[str, int] = {}
    for c in clients:
        profile_counts[c.profile] = profile_counts.get(c.profile, 0) + 1
    return {
        "num_clients": len(clients),
        "total_records": sum(len(c.records) for c in clients),
        "profile_distribution": profile_counts,
        "sample_client": clients[0].to_metadata() if clients else None,
    }


@app.get("/api/v1/datasets/real-hdfs/summary")
def real_hdfs_summary():
    """Summary of real LogHub HDFS telemetry used for production-log scenarios."""
    from app.simulation.real_telemetry_loader import get_real_data_summary

    return get_real_data_summary()


@app.post("/api/v1/simulate/non-iid")
def simulate_non_iid(request: SimulateNonIIDRequest):
    profiles = assign_client_profiles(
        request.num_clients, request.severity, request.random_seed
    )
    counts: dict[str, int] = {}
    for p in profiles:
        counts[p] = counts.get(p, 0) + 1
    return {"severity": request.severity, "profile_distribution": counts}
