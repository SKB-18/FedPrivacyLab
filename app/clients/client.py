"""Base federated client interface."""

import hashlib
from abc import ABC, abstractmethod
from typing import Any

import numpy as np


def hash_client_id(client_id: str) -> str:
    return hashlib.sha256(client_id.encode()).hexdigest()[:16]


class BaseClient(ABC):
    def __init__(self, client_id: str):
        self.client_id = client_id
        self.id_hash = hash_client_id(client_id)

    @abstractmethod
    def get_local_records(self) -> list[dict]:
        pass

    @abstractmethod
    def num_examples(self) -> int:
        pass

    @property
    def metadata(self) -> dict[str, Any]:
        return {"client_id_hash": self.id_hash, "num_examples": self.num_examples()}
