"""SQLite store of calibration samples.

Each row is one approved alignment:
  (world_dot, eye-corner geometry)  ->  (display pixel you approved)
captured from the high-speed eye snapshot at the instant you pressed Approve.

Depends only on numpy + stdlib.
"""
import os
import json
import sqlite3
import time
from typing import Tuple
import numpy as np


class Dataset:
    def __init__(self, db_path: str, feature_names):
        self.db_path = db_path
        self.feature_names = list(feature_names)
        d = os.path.dirname(db_path)
        if d:
            os.makedirs(d, exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self._init_schema()

    def _init_schema(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS samples (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                ts         REAL NOT NULL,
                features   TEXT NOT NULL,   -- json list of floats
                pixel_x    REAL NOT NULL,   -- normalized [0,1]
                pixel_y    REAL NOT NULL,   -- normalized [0,1]
                weight     REAL DEFAULT 1.0,-- eye-tracker confidence for this sample
                note       TEXT
            )
            """
        )
        # migrate older DBs that predate the weight column
        cols = [r[1] for r in self.conn.execute("PRAGMA table_info(samples)")]
        if "weight" not in cols:
            self.conn.execute("ALTER TABLE samples ADD COLUMN weight REAL DEFAULT 1.0")
        self.conn.commit()

    def add(self, features, pixel, weight: float = 1.0, note: str = "") -> int:
        features = [float(v) for v in features]
        cur = self.conn.execute(
            "INSERT INTO samples (ts, features, pixel_x, pixel_y, weight, note) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (time.time(), json.dumps(features), float(pixel[0]),
             float(pixel[1]), float(weight), note),
        )
        self.conn.commit()
        return cur.lastrowid

    def load(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        rows = self.conn.execute(
            "SELECT features, pixel_x, pixel_y, COALESCE(weight, 1.0) "
            "FROM samples ORDER BY id"
        ).fetchall()
        if not rows:
            n = len(self.feature_names)
            return np.empty((0, n)), np.empty((0, 2)), np.empty((0,))
        X = np.array([json.loads(r[0]) for r in rows], dtype=float)
        Y = np.array([[r[1], r[2]] for r in rows], dtype=float)
        W = np.array([r[3] for r in rows], dtype=float)
        return X, Y, W

    def count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM samples").fetchone()[0]

    def undo_last(self):
        """UNDO fallback: delete the most recently added sample (a mis-approved/mis-nudged
        one) and return it as (features, pixel, weight), or None if the DB is empty. The
        live loop retrains after this, so a bad point can be removed without restarting."""
        row = self.conn.execute(
            "SELECT id, features, pixel_x, pixel_y, COALESCE(weight, 1.0) "
            "FROM samples ORDER BY id DESC LIMIT 1").fetchone()
        if row is None:
            return None
        self.conn.execute("DELETE FROM samples WHERE id = ?", (row[0],))
        self.conn.commit()
        return (np.array(json.loads(row[1]), float),
                np.array([row[2], row[3]], float), float(row[4]))

    def clear(self) -> int:
        """RESET fallback: wipe every stored sample (start the calibration over). Returns how
        many rows were removed. Destructive — the live loop guards it behind a confirm key /
        the --reset-db CLI flag so misinputted data can be cleared deliberately, not by accident."""
        n = self.count()
        self.conn.execute("DELETE FROM samples")
        self.conn.commit()
        return n

    def close(self) -> None:
        self.conn.close()
