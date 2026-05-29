"""Tests for real HDFS telemetry loader."""

from pathlib import Path

import pandas as pd
import pytest

from app.simulation.real_telemetry_loader import HDFS_CACHE, _parse_hdfs_to_records


@pytest.fixture
def mini_hdfs_csv(tmp_path, monkeypatch):
    sample = tmp_path / "hdfs_mini.csv"
    sample.write_text(
        "LineId,Date,Time,Pid,Level,Component,Content,EventId,EventTemplate\n"
        "1,081109,203615,1,INFO,dfs.FSNamesystem,Receiving block,1,Receiving block src\n"
        "2,081109,203616,2,ERROR,dfs.DataNode$PacketResponder,Failed to transfer,2,Failed to transfer\n"
        "3,081109,203617,3,INFO,dfs.DataNode$PacketResponder,Received block,3,Received block\n"
        "4,081109,203620,4,WARN,dfs.FSNamesystem,Slow response,4,Slow response\n"
        "5,081109,203625,5,INFO,dfs.DataNode$DataXceiver,PacketResponder done,5,PacketResponder done\n"
        "6,081109,203630,6,INFO,dfs.DataNode$DataXceiver,Received block,6,Received block\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "app.simulation.real_telemetry_loader.HDFS_CACHE", sample
    )
    return sample


def test_parse_hdfs_to_records(mini_hdfs_csv):
    df = pd.read_csv(mini_hdfs_csv)
    clients = _parse_hdfs_to_records(df)
    assert len(clients) >= 2
    assert all(len(v) >= 1 for v in clients.values())
    first = next(iter(clients.values()))[0]
    assert "latency_ms" in first
    assert "label_poor_experience" in first


def test_real_data_summary_download():
    """Integration: download real file if network available."""
    if HDFS_CACHE.exists() and HDFS_CACHE.stat().st_size > 1000:
        from app.simulation.real_telemetry_loader import get_real_data_summary

        s = get_real_data_summary()
        assert s["num_clients"] > 0
        assert s["total_events"] > 0
