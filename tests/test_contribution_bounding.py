"""Unit tests for contribution bounding."""

from app.privacy.contribution_bounding import bound_analytics_summary


def test_bounds_applied():
    result = bound_analytics_summary(
        failure_count=100,
        event_count=1000,
        latency_sum=999999,
        latency_count=1000,
        click_count=500,
        bounds={
            "max_events_per_client_per_round": 20,
            "max_failures_per_client_per_round": 5,
            "max_latency_sum_per_client": 50000,
        },
    )
    assert result.failure_count == 5
    assert result.event_count == 20
    assert result.latency_sum == 50000
    assert result.bounded is True


def test_within_bounds_unchanged():
    result = bound_analytics_summary(1, 5, 1000.0, 5, 2)
    assert result.failure_count == 1
    assert result.event_count == 5
    assert result.bounded is False
