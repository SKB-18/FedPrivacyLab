"""Federated analytics API routes."""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import AnalyticsResult
from app.schemas import AnalyticsResultResponse

router = APIRouter(prefix="/api/v1/experiments", tags=["analytics"])


@router.get("/{experiment_id}/analytics", response_model=list[AnalyticsResultResponse])
def get_analytics(experiment_id: int, db: Session = Depends(get_db)):
    rows = (
        db.query(AnalyticsResult)
        .filter(AnalyticsResult.experiment_id == experiment_id)
        .order_by(AnalyticsResult.round_number, AnalyticsResult.metric_name)
        .all()
    )
    if not rows:
        return []
    return rows
