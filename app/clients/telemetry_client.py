"""Telemetry federated client with local data."""

from app.clients.client import BaseClient
from app.simulation.synthetic_telemetry import SyntheticClient


class TelemetryClient(BaseClient):
    def __init__(self, synthetic: SyntheticClient):
        super().__init__(synthetic.client_id)
        self._synthetic = synthetic
        self.profile = synthetic.profile

    def get_local_records(self) -> list[dict]:
        return [r.to_dict() for r in self._synthetic.records]

    def num_examples(self) -> int:
        return len(self._synthetic.records)

    @property
    def cohort(self) -> str:
        return f"{self._synthetic.locale}_{self._synthetic.device_tier}"
