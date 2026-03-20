from unittest.mock import Mock, patch
import pytest

from src.kafka_consumers.fraud_listener import FraudListener

@pytest.fixture
def mock_config_loader():
    config = Mock()
    config.config = {
        "api": {
            "fraud_detection": {
                "kafka": {
                    "listener_toggle": True,
                    "topic": "test_input_topic"
                }
            }
        }
    }
    return config

@pytest.fixture
def mock_config_loader_toggle_off():
    config = Mock()
    config.config = {
        "api": {
            "fraud_detection": {
                "kafka": {
                    "listener_toggle": False,
                    "topic": "test_input_topic"
                }
            }
        }
    }
    return config

@patch("src.kafka_consumers.fraud_listener.KafkaConfigLoader")
@patch("src.kafka_consumers.fraud_listener.KafkaListener")
def test_start_listener_toggle_on(mock_kafka_listener, mock_kafka_config_loader, mock_config_loader):
    listener = FraudListener(mock_config_loader)
    handler = Mock()

    listener.start_listener(handler)

    mock_kafka_config_loader.assert_called_once_with(mock_config_loader)
    mock_kafka_listener.assert_called_once_with("test_input_topic", handler, mock_kafka_config_loader.return_value)
    mock_kafka_listener.return_value.start.assert_called_once()

@patch("src.kafka_consumers.fraud_listener.KafkaConfigLoader")
@patch("src.kafka_consumers.fraud_listener.KafkaListener")
def test_start_listener_toggle_off(mock_kafka_listener, mock_kafka_config_loader, mock_config_loader_toggle_off):
    listener = FraudListener(mock_config_loader_toggle_off)
    handler = Mock()

    listener.start_listener(handler)

    mock_kafka_config_loader.assert_not_called()
    mock_kafka_listener.assert_not_called()

@patch("src.kafka_consumers.fraud_listener.KafkaConfigLoader")
@patch("src.kafka_consumers.fraud_listener.KafkaListener")
def test_stop_listener_calls_stop(mock_kafka_listener, mock_kafka_config_loader, mock_config_loader):
    listener = FraudListener(mock_config_loader)
    handler = Mock()

    listener.start_listener(handler)
    listener.stop_listener()

    mock_kafka_listener.return_value.stop.assert_called_once()

def test_stop_listener_before_start_is_safe(mock_config_loader):
    listener = FraudListener(mock_config_loader)
    # Should not raise even if start_listener was never called
    listener.stop_listener()
