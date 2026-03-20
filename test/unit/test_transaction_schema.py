from datetime import datetime
from uuid import UUID

import pytest
from pydantic import ValidationError

from src.schemas.transaction import TransactionBase, TransactionCanonical

def test_transaction_base_valid():
    data = {
        "event_time_seconds": 1600000000,
        "amount": 100.5,
        "features": {"f1": 0.1, "f2": 0.2},
        "is_fraud": False,
        "data_source": "test_source"
    }
    obj = TransactionBase(**data)
    assert isinstance(obj.transaction_id, UUID)
    assert obj.event_time_seconds == 1600000000
    assert obj.amount == 100.5
    assert obj.features == {"f1": 0.1, "f2": 0.2}
    assert obj.is_fraud is False
    assert obj.data_source == "test_source"

def test_transaction_base_invalid_amount():
    data = {
        "event_time_seconds": 1600000000,
        "amount": -10.0,
        "features": {},
        "data_source": "test_source"
    }
    with pytest.raises(ValidationError):
        TransactionBase(**data)

def test_transaction_canonical_with_string_timestamp():
    data = {
        "event_time_seconds": 1600000000,
        "amount": 50.0,
        "features": {},
        "data_source": "test_source",
        "event_timestamp": "2023-01-01T12:00:00"
    }
    obj = TransactionCanonical(**data)
    assert isinstance(obj.event_timestamp, datetime)
    assert obj.event_timestamp.year == 2023
    assert obj.event_timestamp.month == 1
    assert obj.event_timestamp.day == 1

def test_transaction_canonical_with_datetime_timestamp():
    dt = datetime(2023, 1, 1, 12, 0, 0)
    data = {
        "event_time_seconds": 1600000000,
        "amount": 50.0,
        "features": {},
        "data_source": "test_source",
        "event_timestamp": dt
    }
    obj = TransactionCanonical(**data)
    assert obj.event_timestamp == dt
