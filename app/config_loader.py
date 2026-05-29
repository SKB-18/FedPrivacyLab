"""Load YAML configuration files."""

from pathlib import Path
from typing import Any

import yaml

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"

# Set per experiment run (coordinator sets this from dataset name)
_active_dataset: str | None = None


def set_active_dataset(dataset: str | None) -> None:
    global _active_dataset
    _active_dataset = dataset


def load_yaml(name: str) -> dict[str, Any]:
    path = CONFIG_DIR / name
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def get_privacy_bounds(dataset: str | None = None) -> dict[str, Any]:
    cfg = load_yaml("privacy_config.yaml")
    bounds = dict(cfg.get("contribution_bounds", {}))
    ds = dataset or _active_dataset
    if ds:
        overrides = cfg.get("dataset_overrides", {}).get(ds, {})
        bounds.update({k: v for k, v in overrides.items() if k.startswith("max_")})
    # Proportional cap: sum bound scales with events bound
    max_events = int(bounds.get("max_events_per_client_per_round", 20))
    per_event = float(bounds.get("max_latency_ms_per_event", 5000))
    bounds["max_latency_sum_per_client"] = min(
        float(bounds.get("max_latency_sum_per_client", max_events * per_event)),
        max_events * per_event,
    )
    return bounds


def get_cohort_threshold(dataset: str | None = None) -> int:
    cfg = load_yaml("privacy_config.yaml")
    ds = dataset or _active_dataset
    if ds:
        override = cfg.get("dataset_overrides", {}).get(ds, {}).get(
            "min_clients_for_reporting"
        )
        if override is not None:
            return int(override)
    return int(cfg.get("cohort", {}).get("min_clients_for_reporting", 100))
