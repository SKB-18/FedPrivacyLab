"""Non-IID client profile assignment."""

import numpy as np

CLIENT_PROFILES = [
    "fast_wifi",
    "slow_cellular",
    "old_app_version",
    "high_failure",
    "power_user",
    "low_engagement",
]

PROFILE_PARAMS = {
    "fast_wifi": {
        "latency_mean": 400,
        "latency_std": 100,
        "failure_prob": 0.02,
        "click_prob": 0.35,
        "session_mean": 240,
        "events_mult": 1.0,
    },
    "slow_cellular": {
        "latency_mean": 1800,
        "latency_std": 400,
        "failure_prob": 0.12,
        "click_prob": 0.15,
        "session_mean": 120,
        "events_mult": 0.8,
    },
    "old_app_version": {
        "latency_mean": 1100,
        "latency_std": 300,
        "failure_prob": 0.18,
        "click_prob": 0.2,
        "session_mean": 150,
        "events_mult": 1.0,
    },
    "high_failure": {
        "latency_mean": 1400,
        "latency_std": 350,
        "failure_prob": 0.35,
        "click_prob": 0.1,
        "session_mean": 90,
        "events_mult": 1.2,
    },
    "power_user": {
        "latency_mean": 700,
        "latency_std": 200,
        "failure_prob": 0.05,
        "click_prob": 0.5,
        "session_mean": 360,
        "events_mult": 2.0,
    },
    "low_engagement": {
        "latency_mean": 900,
        "latency_std": 250,
        "failure_prob": 0.08,
        "click_prob": 0.05,
        "session_mean": 45,
        "events_mult": 0.5,
    },
}


def assign_client_profiles(
    num_clients: int, severity: float = 1.0, seed: int = 42
) -> list[str]:
    """Assign non-IID profiles to clients with optional severity skew."""
    rng = np.random.default_rng(seed)
    profiles = list(CLIENT_PROFILES)
    if severity > 1.0:
        weights = np.array([1.0, severity, severity, severity * 1.2, 1.0, severity])
        weights = weights / weights.sum()
        return list(rng.choice(profiles, size=num_clients, p=weights))
    return list(rng.choice(profiles, size=num_clients))


def get_profile_params(profile: str) -> dict:
    return PROFILE_PARAMS.get(profile, PROFILE_PARAMS["fast_wifi"])
