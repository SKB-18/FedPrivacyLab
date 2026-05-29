"""Download and verify real telemetry dataset (LogHub HDFS)."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.simulation.real_telemetry_loader import ensure_hdfs_dataset, get_real_data_summary


def main():
    path = ensure_hdfs_dataset()
    print(f"Cached at: {path}")
    summary = get_real_data_summary()
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
