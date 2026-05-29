"""Federated learning API routes."""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import RoundMetric

router = APIRouter(prefix="/api/v1/learning", tags=["learning"])


@router.get("/experiments/{experiment_id}/summary")
def learning_summary(experiment_id: int, db: Session = Depends(get_db)):
    rows = (
        db.query(RoundMetric)
        .filter(RoundMetric.experiment_id == experiment_id)
        .order_by(RoundMetric.round_number)
        .all()
    )
    return {
        "experiment_id": experiment_id,
        "rounds": [
            {
                "round": r.round_number,
                "accuracy": r.accuracy,
                "roc_auc": r.roc_auc,
                "f1": r.f1_score,
                "loss": r.eval_loss,
            }
            for r in rows
        ],
    }
