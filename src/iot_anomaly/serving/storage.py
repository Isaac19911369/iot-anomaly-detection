"""Optional persistence of scores to TimescaleDB / PostgreSQL."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import datetime

log = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS anomaly_scores (
    time        TIMESTAMPTZ      NOT NULL,
    device_id   TEXT             NOT NULL,
    score       DOUBLE PRECISION NOT NULL,
    is_anomaly  BOOLEAN          NOT NULL,
    model       TEXT             NOT NULL
);
"""
HYPERTABLE = "SELECT create_hypertable('anomaly_scores', 'time', if_not_exists => TRUE);"


class ScoreStore:
    """Writes scored points; silently disabled when no DATABASE_URL is configured."""

    def __init__(self, dsn: str | None):
        self.dsn = dsn
        self.enabled = bool(dsn)
        if self.enabled:
            self._init_schema()

    def _connect(self):
        import psycopg

        return psycopg.connect(self.dsn, autocommit=True)

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(SCHEMA)
            try:
                conn.execute(HYPERTABLE)
            except Exception as exc:  # plain PostgreSQL without the extension
                log.info("TimescaleDB hypertable not created: %s", exc)

    def write(
        self,
        device_id: str,
        model: str,
        times: Sequence[datetime],
        scores: Sequence[float],
        flags: Sequence[bool],
    ) -> int:
        if not self.enabled or not times:
            return 0
        rows = [
            (t, device_id, float(s), bool(f), model)
            for t, s, f in zip(times, scores, flags, strict=True)
        ]
        with self._connect() as conn, conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO anomaly_scores (time, device_id, score, is_anomaly, model) "
                "VALUES (%s, %s, %s, %s, %s)",
                rows,
            )
        return len(rows)
