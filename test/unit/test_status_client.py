from unittest.mock import MagicMock, Mock, patch

import pytest

from src.clients.status_client import TransactionStatusClient


@pytest.fixture
def mock_config_loader():
    loader = Mock()
    loader.config = {"database": {"url": "postgresql://user:pass@localhost:5432/db"}}
    return loader


@pytest.fixture
@patch("src.clients.status_client.create_engine")
def status_client(mock_create_engine, mock_config_loader):
    mock_engine = MagicMock()
    mock_create_engine.return_value = mock_engine
    client = TransactionStatusClient(mock_config_loader)
    return client, mock_engine


def test_mark_received_upserts_all_ids(status_client):
    client, mock_engine = status_client
    mock_conn = MagicMock()
    mock_engine.begin.return_value.__enter__.return_value = mock_conn

    client.mark_received(["id-1", "id-2"])

    mock_conn.execute.assert_called_once()
    stmt, params = mock_conn.execute.call_args[0]
    assert "received_at" in str(stmt)
    assert params == [
        {"transaction_id": "id-1", "status": "received", "ts": params[0]["ts"]},
        {"transaction_id": "id-2", "status": "received", "ts": params[0]["ts"]},
    ]


def test_mark_flagged_upserts_all_ids(status_client):
    client, mock_engine = status_client
    mock_conn = MagicMock()
    mock_engine.begin.return_value.__enter__.return_value = mock_conn

    client.mark_flagged(["id-1"])

    stmt, params = mock_conn.execute.call_args[0]
    assert "flagged_at" in str(stmt)
    assert params[0]["status"] == "flagged"


def test_mark_many_empty_list_is_a_noop(status_client):
    client, mock_engine = status_client

    client.mark_received([])

    mock_engine.begin.assert_not_called()


def test_mark_many_swallows_db_errors(status_client):
    # A Postgres hiccup on status tracking must never bubble up and crash the
    # Kafka consumer / block score publishing.
    client, mock_engine = status_client
    mock_engine.begin.side_effect = RuntimeError("connection refused")

    client.mark_received(["id-1"])  # should not raise
