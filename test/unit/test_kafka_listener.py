from unittest.mock import Mock
import pytest

from src.kafka_consumers.kafka_listener import KafkaListener

@pytest.fixture
def mock_kafka_config_loader():
    loader = Mock()
    loader.consumer = Mock()
    return loader

def test_start_consumes_messages(mock_kafka_config_loader):
    handler = Mock()
    listener = KafkaListener("test_topic", handler, mock_kafka_config_loader)
    
    mock_msg_success = Mock()
    mock_msg_success.error.return_value = None
    mock_msg_success.value.return_value = b'{"data": 1}'
    
    mock_msg_error = Mock()
    mock_msg_error.error.return_value = True
    
    mock_kafka_config_loader.consumer.poll.side_effect = [
        mock_msg_success,
        mock_msg_error,
        None,
        Exception("Stop Loop")
    ]
    
    with pytest.raises(Exception, match="Stop Loop"):
        listener.start()

    listener.consumer.subscribe.assert_called_once_with(["test_topic"])
    handler.assert_called_once_with(b'{"data": 1}')
    listener.consumer.commit.assert_called_once_with(mock_msg_success)
    assert listener.consumer.poll.call_count == 4
    # The consumer is closed in start()'s finally, on the polling thread.
    listener.consumer.close.assert_called_once()

def test_stop_signals_only_does_not_close(mock_kafka_config_loader):
    # stop() must NOT close the consumer (that happens on the polling thread);
    # calling close() from another thread races an in-flight poll().
    listener = KafkaListener("test_topic", Mock(), mock_kafka_config_loader)
    listener.stop()
    assert listener._stop_event.is_set()
    listener.consumer.close.assert_not_called()

def test_start_closes_consumer_when_stop_event_set(mock_kafka_config_loader):
    listener = KafkaListener("test_topic", Mock(), mock_kafka_config_loader)
    listener._stop_event.set()
    listener.start()
    listener.consumer.close.assert_called_once()
