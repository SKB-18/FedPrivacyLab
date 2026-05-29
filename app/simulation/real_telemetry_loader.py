"""
Load real distributed-systems telemetry from LogHub HDFS structured logs.

Source: LogPAI LogHub — HDFS_2k.log_structured.csv (production Hadoop logs).
Each HDFS component (e.g., NameNode, DataNode) is treated as a federated client;
events map to app-style telemetry for privacy-preserving measurement scenarios.

Reference: https://github.com/logpai/loghub/tree/master/HDFS
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any
from urllib.request import urlretrieve

import numpy as np
import pandas as pd

from app.simulation.synthetic_telemetry import (
    SyntheticClient,
    TelemetryRecord,
    _compute_poor_experience_label,
)

RAW_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "raw"
HDFS_URL = (
    "https://raw.githubusercontent.com/logpai/loghub/master/HDFS/HDFS_2k.log_structured.csv"
)
HDFS_CACHE = RAW_DIR / "HDFS_2k.log_structured.csv"

# Map HDFS components to product-style feature names (real scenario narrative)
COMPONENT_TO_FEATURE = {
    "NameNode": "metadata_service",
    "DataNode": "storage_io",
    "E10": "writing_tools",
    "E11": "visual_search",
    "E12": "dictation",
    "E13": "translation",
    "E14": "spotlight",
    "IPC": "ipc_channel",
    "PacketResponder": "network_stack",
}

DEVICE_BY_COMPONENT = {
    "NameNode": "high",
    "DataNode": "mid",
    "PacketResponder": "low",
}


def ensure_hdfs_dataset() -> Path:
    """Download HDFS log sample if not cached."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if HDFS_CACHE.exists() and HDFS_CACHE.stat().st_size > 1000:
        return HDFS_CACHE
    print(f"Downloading real HDFS telemetry from LogHub -> {HDFS_CACHE}")
    urlretrieve(HDFS_URL, HDFS_CACHE)
    return HDFS_CACHE


def _normalize_component(component: str) -> str:
    """e.g. dfs.DataNode$PacketResponder -> PacketResponder"""
    if "$" in component:
        return component.split("$")[-1]
    if "." in component:
        return component.split(".")[-1]
    return component


def _component_feature(component: str) -> str:
    short = _normalize_component(component)
    if short in COMPONENT_TO_FEATURE:
        return COMPONENT_TO_FEATURE[short]
    if component in COMPONENT_TO_FEATURE:
        return COMPONENT_TO_FEATURE[component]
    h = int(hashlib.md5(component.encode()).hexdigest()[:8], 16)
    features = list(COMPONENT_TO_FEATURE.values())
    return features[h % len(features)]


def _parse_hdfs_to_records(
    df: pd.DataFrame, partition_by: str = "pid"
) -> dict[str, list[dict]]:
    """Partition log events by Pid (many clients) or Component (few clients)."""
    df = df.copy()
    date_str = df["Date"].astype(str).str.zfill(6)
    time_str = df["Time"].astype(str).str.zfill(6)
    df["datetime"] = pd.to_datetime(
        date_str + time_str,
        format="%y%m%d%H%M%S",
        errors="coerce",
    )
    if df["datetime"].isna().all():
        df["datetime"] = pd.to_datetime(
            df["Date"].astype(str) + " " + df["Time"].astype(str),
            errors="coerce",
        )
    df = df.dropna(subset=["datetime"]).sort_values("datetime")

    clients: dict[str, list[dict]] = {}

    group_col = "Pid" if partition_by == "pid" and "Pid" in df.columns else "Component"

    for group_key, group in df.groupby(group_col):
        if group_col == "Pid":
            component = str(group["Component"].iloc[0])
            short = _normalize_component(component)
            client_id = f"hdfs_pid_{group_key}"
        else:
            component = str(group_key)
            short = _normalize_component(component)
            client_id = f"hdfs_{short}"
        group = group.sort_values("datetime")
        prev_ts = None
        records: list[dict] = []

        for _, row in group.iterrows():
            ts = row["datetime"]
            if prev_ts is not None:
                delta_ms = min(5000.0, max(50.0, (ts - prev_ts).total_seconds() * 1000))
            else:
                delta_ms = 200.0
            prev_ts = ts

            level = str(row.get("Level", "INFO"))
            failure = 1 if level in ("ERROR", "WARN") else 0
            # Success-path events as engagement proxy (real ops pattern)
            content = str(row.get("EventTemplate", ""))
            clicked = 1 if any(
                k in content.lower() for k in ("receive", "add", "success", "complete")
            ) else 0

            session_sec = min(600.0, delta_ms / 10.0 + 30.0)
            latency_ms = delta_ms + (800.0 if failure else 0.0)

            # Log inter-arrival gaps are seconds-scale; cap only for label thresholds so
            # poor_experience_rate is not ~90% (which broke analytics charts and FL labels).
            label_latency_ms = min(latency_ms, 2000.0)
            poor = _compute_poor_experience_label(
                label_latency_ms, failure, clicked, session_sec
            )

            feature = _component_feature(str(component))
            device = DEVICE_BY_COMPONENT.get(short, DEVICE_BY_COMPONENT.get(str(component), "mid"))
            locale = "en_US" if "NameNode" in short or "FSNamesystem" in short else "en_GB"

            records.append(
                {
                    "feature": feature,
                    "latency_ms": float(latency_ms),
                    "failure": failure,
                    "clicked_recommendation": clicked,
                    "session_length_sec": float(session_sec),
                    "device_tier": device,
                    "network_type": "wifi" if device == "high" else "cellular",
                    "app_version": "1.3",
                    "locale": locale,
                    "locale_bucket": locale,
                    "label_poor_experience": poor,
                    "source": "loghub_hdfs",
                    "raw_component": str(component),
                    "raw_level": level,
                }
            )
        if records:
            clients[client_id] = records

    return clients


def load_real_hdfs_clients(
    max_clients: int | None = None,
    min_records_per_client: int = 3,
    partition_by: str = "component",
) -> list[SyntheticClient]:
    """
    Build federated clients from real HDFS production logs.
    Default partition: HDFS subsystem (Component) — realistic service-level clients
    with enough events per client for FL and analytics.
    Use partition_by='pid' for fine-grained scale experiments (many sparse clients).
    """
    path = ensure_hdfs_dataset()
    df = pd.read_csv(path)
    client_map = _parse_hdfs_to_records(df, partition_by=partition_by)

    clients: list[SyntheticClient] = []
    for client_id, records in sorted(client_map.items()):
        if len(records) < min_records_per_client:
            continue
        rec_objs = [
            TelemetryRecord(
                feature=r["feature"],
                latency_ms=r["latency_ms"],
                failure=r["failure"],
                clicked_recommendation=r["clicked_recommendation"],
                session_length_sec=r["session_length_sec"],
                device_tier=r["device_tier"],
                network_type=r["network_type"],
                app_version=r["app_version"],
                locale=r["locale"],
                label_poor_experience=r["label_poor_experience"],
            )
            for r in records
        ]
        profile = "high_failure" if sum(r.failure for r in rec_objs) / len(rec_objs) > 0.15 else "fast_wifi"
        clients.append(
            SyntheticClient(
                client_id=client_id,
                profile=profile,
                locale=rec_objs[0].locale,
                app_version=rec_objs[0].app_version,
                device_tier=rec_objs[0].device_tier,
                network_type=rec_objs[0].network_type,
                records=rec_objs,
            )
        )

    if max_clients and len(clients) > max_clients:
        # Prefer clients with more events and failures (representative sample)
        clients.sort(
            key=lambda c: (
                sum(r.failure for r in c.records),
                len(c.records),
            ),
            reverse=True,
        )
        clients = clients[:max_clients]

    return clients


def get_real_data_summary(partition_by: str = "component") -> dict[str, Any]:
    """Summary stats for dashboard (real-life scenario context)."""
    clients = load_real_hdfs_clients(partition_by=partition_by)
    total_records = sum(len(c.records) for c in clients)
    failure_rate = (
        sum(r.failure for c in clients for r in c.records) / max(total_records, 1)
    )
    avg_latency = (
        sum(r.latency_ms for c in clients for r in c.records) / max(total_records, 1)
    )
    return {
        "dataset": "real_hdfs_loghub",
        "source_url": HDFS_URL,
        "num_clients": len(clients),
        "total_events": total_records,
        "failure_rate": round(failure_rate, 4),
        "avg_latency_ms": round(avg_latency, 2),
        "description": (
            "Production HDFS logs (LogPAI LogHub); federated clients = HDFS subsystems "
            "(component partition) or processes (pid partition)."
        ),
        "partition": partition_by,
    }
