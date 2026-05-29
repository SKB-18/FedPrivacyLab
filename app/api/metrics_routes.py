"""Metrics API routes."""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import RoundMetric
from app.schemas import RoundMetricResponse

router = APIRouter(prefix="/api/v1/experiments", tags=["metrics"])


@router.get("/{experiment_id}/metrics", response_model=list[RoundMetricResponse])
def get_metrics(experiment_id: int, db: Session = Depends(get_db)):
    rows = (
        db.query(RoundMetric)
        .filter(RoundMetric.experiment_id == experiment_id)
        .order_by(RoundMetric.round_number)
        .all()
    )
    if not rows:
        return []
    return rows
