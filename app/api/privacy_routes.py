"""Privacy report API routes."""

import json

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import Experiment, PrivacyReport
from app.privacy.risk_report import build_privacy_report
from app.schemas import ExperimentStartRequest, PrivacyReportResponse

router = APIRouter(prefix="/api/v1/experiments", tags=["privacy"])


@router.get("/{experiment_id}/privacy-report", response_model=PrivacyReportResponse)
def get_privacy_report(experiment_id: int, db: Session = Depends(get_db)):
    exp = db.query(Experiment).filter(Experiment.id == experiment_id).first()
    if not exp:
        raise HTTPException(status_code=404, detail="Experiment not found")

    latest = (
        db.query(PrivacyReport)
        .filter(PrivacyReport.experiment_id == experiment_id)
        .order_by(PrivacyReport.id.desc())
        .first()
    )

    config = ExperimentStartRequest(
        name=exp.name,
        mode=exp.mode,  # type: ignore
        dataset=exp.dataset,
        num_clients=exp.num_clients or 500,
        rounds=exp.rounds or 10,
        clients_per_round=exp.clients_per_round or 50,
        local_epochs=exp.local_epochs or 2,
        dropout_rate=exp.dropout_rate or 0.15,
        secure_agg_enabled=bool(exp.secure_agg_enabled),
        dp_enabled=bool(exp.dp_enabled),
        clipping_norm=exp.clipping_norm or 1.0,
        noise_multiplier=exp.noise_multiplier or 0.5,
        epsilon=exp.epsilon or 2.0,
        delta=exp.delta or 1e-6,
    )
    report = build_privacy_report(experiment_id, config)
    if latest and latest.notes:
        try:
            report.notes = json.loads(latest.notes)
        except json.JSONDecodeError:
            pass
    return report


@router.get("/{experiment_id}/client-participation")
def get_client_participation(experiment_id: int, db: Session = Depends(get_db)):
    from app.models import ClientParticipation
    from app.schemas import ClientParticipationResponse

    rows = (
        db.query(ClientParticipation)
        .filter(ClientParticipation.experiment_id == experiment_id)
        .order_by(ClientParticipation.round_number)
        .all()
    )
    return [ClientParticipationResponse.model_validate(r) for r in rows]
