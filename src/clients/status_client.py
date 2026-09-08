import logging
from datetime import datetime, timezone

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from config.config_loader import ConfigLoader

logger = logging.getLogger(__name__)

# Whitelisted so the timestamp column can be safely interpolated into the SQL
# (bind params can't parameterize identifiers). Only reached via the two
# mark_*() methods below, never with caller-supplied input.
_STATUS_COLUMNS = {
    "received": "received_at",
    "flagged": "flagged_at",
}


class TransactionStatusClient:
    """Writes pipeline-progress rows to fraud-rag's `transaction_status` table
    (schema/migrations owned by that repo). This service only ever touches
    this one table — never `transactions`/`transaction_embeddings`.
    """

    def __init__(self, config_loader: ConfigLoader):
        url = config_loader.config["database"]["url"]
        self.engine: Engine = create_engine(url, pool_pre_ping=True)

    def _mark_many(self, transaction_ids: list[str], status: str) -> None:
        if not transaction_ids:
            return
        timestamp_column = _STATUS_COLUMNS[status]
        now = datetime.now(timezone.utc)

        stmt = text(f"""
            INSERT INTO transaction_status (transaction_id, status, {timestamp_column}, updated_at)
            VALUES (:transaction_id, :status, :ts, :ts)
            ON CONFLICT (transaction_id) DO UPDATE SET
                status = EXCLUDED.status,
                {timestamp_column} = EXCLUDED.{timestamp_column},
                updated_at = EXCLUDED.updated_at
        """)
        params = [{"transaction_id": tx_id, "status": status, "ts": now} for tx_id in transaction_ids]
        try:
            with self.engine.begin() as conn:
                conn.execute(stmt, params)
        except Exception:
            # Status tracking is observability, not the core pipeline — a
            # Postgres hiccup here must never block scoring/publishing decisions.
            logger.error("Failed to mark %d transaction(s) as %s", len(transaction_ids), status, exc_info=True)

    def mark_received(self, transaction_ids: list[str]) -> None:
        self._mark_many(transaction_ids, "received")

    def mark_flagged(self, transaction_ids: list[str]) -> None:
        self._mark_many(transaction_ids, "flagged")
