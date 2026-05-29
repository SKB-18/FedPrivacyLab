"""Synthetic client telemetry dataset generator."""

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from app.config_loader import get_privacy_bounds
from app.simulation.non_iid_partition import assign_client_profiles, get_profile_params

FEATURES = ["writing_tools", "visual_search", "dictation", "translation", "spotlight"]
LOCALES = ["en_US", "en_GB", "de_DE", "fr_FR", "ja_JP", "es_ES"]
DEVICE_TIERS = ["low", "mid", "high"]
NETWORK_TYPES = ["wifi", "cellular", "offline"]
APP_VERSIONS = ["1.1", "1.2", "1.3", "1.4"]


@dataclass
class TelemetryRecord:
    feature: str
    latency_ms: float
    failure: int
    clicked_recommendation: int
    session_length_sec: float
    device_tier: str
    network_type: str
    app_version: str
    locale: str
    label_poor_experience: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature": self.feature,
            "latency_ms": self.latency_ms,
            "failure": self.failure,
            "clicked_recommendation": self.clicked_recommendation,
            "session_length_sec": self.session_length_sec,
            "device_tier": self.device_tier,
            "network_type": self.network_type,
            "app_version": self.app_version,
            "locale": self.locale,
            "locale_bucket": self.locale,
            "label_poor_experience": self.label_poor_experience,
        }


@dataclass
class SyntheticClient:
    client_id: str
    profile: str
    locale: str
    app_version: str
    device_tier: str
    network_type: str
    records: list[TelemetryRecord] = field(default_factory=list)

    def to_metadata(self) -> dict[str, Any]:
        return {
            "client_id": self.client_id,
            "profile": self.profile,
            "locale": self.locale,
            "app_version": self.app_version,
            "device_tier": self.device_tier,
            "network_type": self.network_type,
            "num_records": len(self.records),
        }


def _compute_poor_experience_label(
    latency_ms: float, failure: int, clicked: int, session_sec: float
) -> int:
    if failure >= 1 and latency_ms > 1500:
        return 1
    if latency_ms > 3000:
        return 1
    if clicked == 0 and session_sec < 60:
        return 1
    if failure >= 2:
        return 1
    return 0


def generate_client_records(
    profile: str,
    num_records: int,
    rng: np.random.Generator,
    max_latency: float = 5000,
) -> list[TelemetryRecord]:
    params = get_profile_params(profile)
    records = []
    locale = str(rng.choice(LOCALES))
    app_version = str(rng.choice(APP_VERSIONS))
    device_tier = str(rng.choice(DEVICE_TIERS))
    network_type = (
        "wifi" if profile == "fast_wifi" else str(rng.choice(NETWORK_TYPES))
    )

    for _ in range(num_records):
        feature = str(rng.choice(FEATURES))
        latency = float(
            np.clip(
                rng.normal(params["latency_mean"], params["latency_std"]),
                50,
                max_latency,
            )
        )
        failure = int(rng.random() < params["failure_prob"])
        clicked = int(rng.random() < params["click_prob"])
        session = float(max(10, rng.normal(params["session_mean"], 30)))
        label = _compute_poor_experience_label(latency, failure, clicked, session)
        records.append(
            TelemetryRecord(
                feature=feature,
                latency_ms=latency,
                failure=failure,
                clicked_recommendation=clicked,
                session_length_sec=session,
                device_tier=device_tier,
                network_type=network_type,
                app_version=app_version,
                locale=locale,
                label_poor_experience=label,
            )
        )
    return records


def generate_synthetic_clients(
    num_clients: int = 500,
    min_records: int = 5,
    max_records: int = 50,
    seed: int = 42,
    non_iid_severity: float = 1.0,
) -> list[SyntheticClient]:
    """Generate synthetic decentralized clients with local telemetry."""
    rng = np.random.default_rng(seed)
    profiles = assign_client_profiles(num_clients, non_iid_severity, seed)
    bounds = get_privacy_bounds()
    max_latency = float(bounds.get("max_latency_ms_per_event", 5000))
    clients = []

    for i, profile in enumerate(profiles):
        params = get_profile_params(profile)
        base_records = int(rng.integers(min_records, max_records + 1))
        n_records = int(base_records * params["events_mult"])
        n_records = max(min_records, min(n_records, max_records * 2))

        records = generate_client_records(profile, n_records, rng, max_latency)
        client = SyntheticClient(
            client_id=f"client_{i:05d}",
            profile=profile,
            locale=records[0].locale if records else "en_US",
            app_version=records[0].app_version if records else "1.3",
            device_tier=records[0].device_tier if records else "mid",
            network_type=records[0].network_type if records else "wifi",
            records=records,
        )
        clients.append(client)
    return clients
