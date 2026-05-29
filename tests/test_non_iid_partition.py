"""Unit tests for non-IID partitioning."""

from app.simulation.non_iid_partition import (
    CLIENT_PROFILES,
    assign_client_profiles,
    get_profile_params,
)


def test_assign_profiles_count():
    profiles = assign_client_profiles(100, severity=1.0, seed=42)
    assert len(profiles) == 100
    assert all(p in CLIENT_PROFILES for p in profiles)


def test_severity_skews_distribution():
    profiles = assign_client_profiles(1000, severity=1.5, seed=7)
    slow = profiles.count("slow_cellular")
    assert slow > 0


def test_profile_params_exist():
    for p in CLIENT_PROFILES:
        params = get_profile_params(p)
        assert "latency_mean" in params
