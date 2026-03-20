from unittest.mock import Mock, patch
import pytest

from src.services.kafka_service import KafkaService

@pytest.fixture
def mock_kafka_config_loader():
    loader = Mock()
    loader.kafka_producer_config = {"bootstrap.servers": "localhost:9092"}
    loader.kafka_consumer_config = {"bootstrap.servers": "localhost:9092", "group.id": "test"}
    loader.consumer = Mock()
    loader.producer = Mock()
    loader.admin = Mock()
    return loader

@pytest.fixture
def kafka_service(mock_kafka_config_loader):
    return KafkaService(mock_kafka_config_loader)

def test_list_all_topics(kafka_service):
    mock_metadata = Mock()
    mock_metadata.topics = {"topic1": Mock(), "topic2": Mock()}
    kafka_service.admin.list_topics.return_value = mock_metadata
    
    result = kafka_service.list_all_topics()
    
    assert result == ["topic1", "topic2"]
    kafka_service.admin.list_topics.assert_called_once_with(timeout=5)

def test_send_message(kafka_service):
    topic = "test_topic"
    key = "key1"
    message = "message_value"
    
    kafka_service.send_message(topic, key, message)
    
    kafka_service.producer.produce.assert_called_once_with(
        topic,
        key=key,
        value=message,
        callback=kafka_service.delivery_report
    )
    kafka_service.producer.flush.assert_called_once_with(timeout=10)

@patch("src.services.kafka_service.time.time")
def test_consume_topic(mock_time, kafka_service):
    topic = "test_topic"
    mock_time.side_effect = [0, 0, 2]
    
    mock_msg = Mock()
    mock_msg.error.return_value = None
    mock_msg.value.return_value = b'{"msg": "hello"}'
    
    kafka_service.consumer.poll.return_value = mock_msg
    
    result = kafka_service.consume_topic(topic, 1)
    
    assert result == ["{'msg': 'hello'}"]
    kafka_service.consumer.subscribe.assert_called_once_with([topic])
    kafka_service.consumer.close.assert_called_once()

def test_delivery_report_success(kafka_service, caplog):
    mock_msg = Mock()
    mock_msg.topic.return_value = "topic"
    mock_msg.partition.return_value = 0
    mock_msg.offset.return_value = 1
    
    kafka_service.delivery_report(None, mock_msg)
    
    assert "Message delivered to topic partition 0 offset 1" in caplog.text

def test_delivery_report_error(kafka_service, caplog):
    kafka_service.delivery_report("Some error", None)
    
    assert "Message delivery failed: Some error" in caplog.text

@patch("src.services.kafka_service.NewTopic")
def test_create_topic(mock_new_topic, kafka_service):
    topic_name = "new_topic"
    mock_topic_instance = Mock()
    mock_new_topic.return_value = mock_topic_instance
    
    kafka_service.create_topic(topic_name)
    
    mock_new_topic.assert_called_once_with(topic_name, num_partitions=1, replication_factor=1)
    kafka_service.admin.create_topics.assert_called_once_with([mock_topic_instance])

def test_delete_topic(kafka_service):
    topic_name = "old_topic"
    
    kafka_service.delete_topic(topic_name)
    
    kafka_service.admin.delete_topics.assert_called_once_with([topic_name], operation_timeout=10)
