"""Experiment lifecycle API routes."""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.coordinator.experiment_runner import ExperimentRunner
from app.database import get_db
from app.models import Experiment
from app.schemas import ExperimentResponse, ExperimentStartRequest, RunRoundResponse

router = APIRouter(prefix="/api/v1/experiments", tags=["experiments"])


@router.post("/start", response_model=ExperimentResponse)
def start_experiment(request: ExperimentStartRequest, db: Session = Depends(get_db)):
    runner = ExperimentRunner(db)
    exp = runner.create_experiment(request)
    runner.initialize_clients(exp, request)
    return exp


@router.post("/{experiment_id}/run-round", response_model=RunRoundResponse)
def run_round(experiment_id: int, db: Session = Depends(get_db)):
    exp = db.query(Experiment).filter(Experiment.id == experiment_id).first()
    if not exp:
        raise HTTPException(status_code=404, detail="Experiment not found")
    runner = ExperimentRunner(db)
    result = runner.run_single_round(experiment_id)
    return RunRoundResponse(
        experiment_id=experiment_id,
        round_number=result.get("round", 0),
        status="completed",
        metrics=result.get("metrics"),
    )


@router.post("/{experiment_id}/run-all")
def run_all_rounds(experiment_id: int, db: Session = Depends(get_db)):
    exp = db.query(Experiment).filter(Experiment.id == experiment_id).first()
    if not exp:
        raise HTTPException(status_code=404, detail="Experiment not found")
    runner = ExperimentRunner(db)
    return runner.run_full_experiment(experiment_id)


@router.get("/{experiment_id}", response_model=ExperimentResponse)
def get_experiment(experiment_id: int, db: Session = Depends(get_db)):
    exp = db.query(Experiment).filter(Experiment.id == experiment_id).first()
    if not exp:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return exp
