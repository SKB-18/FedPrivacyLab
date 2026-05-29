"""Database helpers for Streamlit dashboard."""

import json
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "fedprivacylab.db"
DATABASE_URL = f"sqlite:///{DB_PATH}"


def get_engine():
    return create_engine(DATABASE_URL, connect_args={"check_same_thread": False})


def load_table(table_name: str) -> pd.DataFrame:
    if not DB_PATH.exists():
        return pd.DataFrame()
    engine = get_engine()
    try:
        return pd.read_sql_table(table_name, engine)
    except Exception:
        return pd.DataFrame()


def get_experiment_ids() -> list[int]:
    df = load_table("experiments")
    if df.empty:
        return []
    return sorted(df["id"].tolist(), reverse=True)


def get_experiment(experiment_id: int) -> dict | None:
    df = load_table("experiments")
    if df.empty:
        return None
    row = df[df["id"] == experiment_id]
    if row.empty:
        return None
    return row.iloc[0].to_dict()


def parse_privacy_notes(notes_val) -> list[str]:
    if notes_val is None or (isinstance(notes_val, float) and pd.isna(notes_val)):
        return []
    if isinstance(notes_val, str):
        try:
            return json.loads(notes_val)
        except json.JSONDecodeError:
            return [notes_val]
    return []
