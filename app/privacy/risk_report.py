"""Privacy risk assessment and reporting."""

from app.schemas import ExperimentStartRequest, PrivacyReportResponse

MODE_LABELS = {
    "centralized_baseline": "Centralized Baseline",
    "federated_analytics": "Federated Analytics",
    "fedavg": "FedAvg",
    "fedavg_secureagg": "FedAvg + Secure Aggregation Simulation",
    "fedavg_secureagg_dp": "FedAvg + Secure Aggregation Simulation + DP",
}


def assess_risk_level(mode: str, dp_enabled: bool, secure_agg: bool) -> str:
    if mode == "centralized_baseline":
        return "high"
    if mode == "fedavg" and not dp_enabled and not secure_agg:
        return "medium-high"
    if mode in ("fedavg_secureagg", "federated_analytics") and not dp_enabled:
        return "medium"
    if dp_enabled:
        return "medium-low"
    return "medium"


def build_privacy_report(
    experiment_id: int,
    config: ExperimentStartRequest | dict,
) -> PrivacyReportResponse:
    if isinstance(config, dict):
        mode = config.get("mode", "fedavg")
        dp = config.get("dp_enabled", False)
        secure = config.get("secure_agg_enabled", False)
        clipping = config.get("clipping_norm", 1.0)
        noise = config.get("noise_multiplier", 0.5)
        epsilon = config.get("epsilon", 2.0)
        delta = config.get("delta", 1e-6)
    else:
        mode = config.mode
        dp = config.dp_enabled
        secure = config.secure_agg_enabled
        clipping = config.clipping_norm
        noise = config.noise_multiplier
        epsilon = config.epsilon
        delta = config.delta

    raw_centralized = mode == "centralized_baseline"
    individual_visible = mode in ("fedavg", "centralized_baseline") and not secure
    clipped = mode not in ("centralized_baseline", "federated_analytics") or mode == "fedavg"

    notes = [
        "Raw client data remains local in federated modes",
        "Secure aggregation is simulated, not production cryptography",
        "DP noise improves privacy but reduces model utility",
        "Client-level privacy accounting is simplified",
        "Synthetic hashed client IDs used only for experiment debugging",
    ]
    if mode == "federated_analytics":
        notes.append("Analytics uses contribution bounding before optional DP noise")
    if dp:
        notes.append("Differential privacy noise applied to aggregates or model updates")

    return PrivacyReportResponse(
        experiment_id=experiment_id,
        privacy_mode=MODE_LABELS.get(mode, mode),
        raw_data_centralized=raw_centralized,
        individual_updates_visible_to_server=individual_visible,
        client_updates_clipped=clipped or mode.startswith("fedavg"),
        dp_noise_added=dp,
        secure_agg_enabled=secure,
        risk_level=assess_risk_level(mode, dp, secure),
        clipping_norm=clipping,
        noise_multiplier=noise,
        epsilon=epsilon,
        delta=delta,
        notes=notes,
    )
