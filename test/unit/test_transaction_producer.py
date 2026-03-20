from unittest.mock import Mock, patch
import pytest

from src.kafka_producers.transaction_producer import TransactionProducer

@pytest.fixture
def mock_kafka_config_loader():
    loader = Mock()
    loader.kafka_producer_config = {
        "burst_mode": False,
        "transactions_per_second": 100
    }
    loader.producer = Mock()
    return loader

@pytest.fixture
def producer(mock_kafka_config_loader):
    data_generator = Mock(return_value={"transaction_id": "tx1", "value": 100})
    return TransactionProducer(
        "test_topic",
        mock_kafka_config_loader,
        data_generator
    )

def test_delivery_callback_success(producer):
    producer._delivery_callback(None, "msg")
    assert producer.stats["total_sent"] == 1
    assert producer.stats["failed"] == 0

def test_delivery_callback_failure(producer):
    producer._delivery_callback("error", "msg")
    assert producer.stats["total_sent"] == 0
    assert producer.stats["failed"] == 1

def test_calculate_delay_no_burst(producer):
    delay = producer._calculate_delay(100.0)
    assert delay == 0.01
    assert producer.stats["current_rate"] == 100

def test_calculate_delay_burst_mode():
    loader = Mock()
    loader.kafka_producer_config = {
        "burst_mode": True,
        "transactions_per_second": 100,
        "burst_interval_seconds": 10,
        "burst_multiplier": 5
    }
    loader.producer = Mock()
    p = TransactionProducer("test", loader, Mock())
    
    delay = p._calculate_delay(5.0)
    assert delay == 1 / 500
    assert p.stats["current_rate"] == 500
    
    delay = p._calculate_delay(15.0)
    assert delay == 1 / 100
    assert p.stats["current_rate"] == 100

@patch("src.kafka_producers.transaction_producer.time.sleep")
@patch("src.kafka_producers.transaction_producer.time.time")
def test_start_loading(mock_time, mock_sleep, producer):
    mock_time.side_effect = [0, 0, 0, 2]
    
    producer.start_loading(duration_seconds=1)
    
    assert producer.data_generator.call_count == 1
    assert producer.producer.produce.call_count == 1
    assert producer.producer.poll.call_count == 1
    assert mock_sleep.call_count == 1
    
    producer.producer.flush.assert_called_once_with(timeout=10)
    assert not producer.running

def test_stop_loading(producer):
    producer.running = True
    producer.stop_loading()
    assert not producer.running
    producer.producer.flush.assert_called_once_with(timeout=10)
