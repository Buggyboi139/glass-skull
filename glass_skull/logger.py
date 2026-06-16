from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import DB_PATH, ensure_dirs


SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    model_name TEXT NOT NULL,
    mode TEXT NOT NULL,
    prompt TEXT NOT NULL,
    output TEXT,
    metadata_json TEXT
);

CREATE TABLE IF NOT EXISTS observations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL,
    layer INTEGER,
    stream TEXT,
    token_index INTEGER,
    token TEXT,
    dimension INTEGER,
    activation REAL,
    abs_activation REAL,
    metadata_json TEXT,
    FOREIGN KEY(run_id) REFERENCES runs(id)
);

CREATE TABLE IF NOT EXISTS edges (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL,
    layer INTEGER,
    module TEXT,
    token_index INTEGER,
    from_dim INTEGER,
    to_dim INTEGER,
    input_activation REAL,
    weight REAL,
    contribution REAL,
    abs_contribution REAL,
    metadata_json TEXT,
    FOREIGN KEY(run_id) REFERENCES runs(id)
);
"""


def connect(path: Path = DB_PATH) -> sqlite3.Connection:
    ensure_dirs()
    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA)
    conn.commit()
    return conn


def log_run(model_name: str, mode: str, prompt: str, output: str | None = None, metadata: dict[str, Any] | None = None) -> int:
    conn = connect()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO runs (created_at, model_name, mode, prompt, output, metadata_json) VALUES (?, ?, ?, ?, ?, ?)",
        (
            datetime.now(timezone.utc).isoformat(),
            model_name,
            mode,
            prompt,
            output,
            json.dumps(metadata or {}),
        ),
    )
    run_id = int(cur.lastrowid)
    conn.commit()
    conn.close()
    return run_id


def log_observations(run_id: int, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    conn = connect()
    cur = conn.cursor()
    cur.executemany(
        """
        INSERT INTO observations
        (run_id, layer, stream, token_index, token, dimension, activation, abs_activation, metadata_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                run_id,
                r.get("layer"),
                r.get("stream"),
                r.get("token_index"),
                r.get("token"),
                r.get("dimension"),
                r.get("activation"),
                r.get("abs_activation"),
                json.dumps(r.get("metadata", {})),
            )
            for r in rows
        ],
    )
    conn.commit()
    conn.close()


def log_edges(run_id: int, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    conn = connect()
    cur = conn.cursor()
    cur.executemany(
        """
        INSERT INTO edges
        (run_id, layer, module, token_index, from_dim, to_dim, input_activation, weight, contribution, abs_contribution, metadata_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                run_id,
                r.get("layer"),
                r.get("module"),
                r.get("token_index"),
                r.get("from_dim"),
                r.get("to_dim"),
                r.get("input_activation"),
                r.get("weight"),
                r.get("contribution"),
                r.get("abs_contribution"),
                json.dumps(r.get("metadata", {})),
            )
            for r in rows
        ],
    )
    conn.commit()
    conn.close()


def recent_runs(limit: int = 25) -> list[dict[str, Any]]:
    conn = connect()
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("SELECT * FROM runs ORDER BY id DESC LIMIT ?", (limit,))
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows
