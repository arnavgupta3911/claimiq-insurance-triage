import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

DB_PATH = Path(__file__).resolve().parent.parent / "claims.db"

COLUMNS = [
    "claim_type",
    "claim_amount",
    "policy_age_months",
    "documents_submitted",
    "documents_required",
    "days_since_incident",
    "previous_claims",
]


def _connect(db_path):
    return sqlite3.connect(db_path or DB_PATH)


def init_db(db_path=None):
    with _connect(db_path) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS claims (
                id INTEGER PRIMARY KEY,
                claim_type TEXT NOT NULL,
                claim_amount REAL NOT NULL,
                policy_age_months INTEGER NOT NULL,
                documents_submitted INTEGER NOT NULL,
                documents_required INTEGER NOT NULL,
                days_since_incident INTEGER NOT NULL,
                previous_claims INTEGER NOT NULL,
                review_score REAL NOT NULL,
                review_queue TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)


def save_claim(claim, result, db_path=None):
    values = [claim[c] for c in COLUMNS] + [
        result["review_score"],
        result["review_queue"],
        datetime.now(timezone.utc).isoformat(timespec="seconds"),
    ]
    placeholders = ", ".join("?" for _ in values)
    columns = ", ".join(COLUMNS + ["review_score", "review_queue", "created_at"])
    with _connect(db_path) as conn:
        cursor = conn.execute(f"INSERT INTO claims ({columns}) VALUES ({placeholders})", values)
        return cursor.lastrowid


def get_claims(db_path=None):
    with _connect(db_path) as conn:
        return pd.read_sql_query("SELECT * FROM claims ORDER BY review_score DESC", conn)
