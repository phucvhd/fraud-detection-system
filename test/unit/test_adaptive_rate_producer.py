from unittest.mock import Mock

import pytest

from src.kafka_producers.adaptive_rate_producer import AdaptiveRateProducer


@pytest.fixture
def mock_kafka_config_loader():
    loader = Mock()
    loader.kafka_producer_config = Mock()
    loader.kafka_producer_config.transactions_per_second = 100
    loader.producer = Mock()
    return loader


@pytest.fixture
def producer(mock_kafka_config_loader):
    data_generator = Mock()
    return AdaptiveRateProducer(
        "test_topic",
        mock_kafka_config_loader,
        data_generator,
        min_rate=10,
        max_rate=1000
    )


def test_adjust_rate_high_lag(producer):
    producer.target_lag_ms = 100
    producer.adjust_rate_based_on_lag(250)
    assert producer.kafka_producer_config.transactions_per_second == 80.0


def test_adjust_rate_high_lag_min_bound(producer):
    producer.target_lag_ms = 100
    producer.kafka_producer_config.transactions_per_second = 11
    producer.adjust_rate_based_on_lag(250)
    assert producer.kafka_producer_config.transactions_per_second == 10.0


def test_adjust_rate_low_lag(producer):
    producer.target_lag_ms = 100
    producer.adjust_rate_based_on_lag(40)
    assert producer.kafka_producer_config.transactions_per_second == 120.0


def test_adjust_rate_low_lag_max_bound(producer):
    producer.target_lag_ms = 100
    producer.kafka_producer_config.transactions_per_second = 900
    producer.adjust_rate_based_on_lag(40)
    assert producer.kafka_producer_config.transactions_per_second == 1000.0


def test_adjust_rate_normal_lag(producer):
    producer.target_lag_ms = 100
    producer.kafka_producer_config.transactions_per_second = 100
    producer.adjust_rate_based_on_lag(100)
    assert producer.kafka_producer_config.transactions_per_second == 100.0
