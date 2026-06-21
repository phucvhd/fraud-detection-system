from unittest.mock import Mock
from fastapi.testclient import TestClient
from fastapi import FastAPI
import pytest


@pytest.fixture(autouse=True)
def mock_dependencies():
    mock_kafka_service = Mock()

    from src.controllers.kafka_controller import router
    app = FastAPI()
    app.state.kafka_service = mock_kafka_service
    app.include_router(router)
    client = TestClient(app)

    yield client, mock_kafka_service


def test_get_all_topics_success(mock_dependencies):
    client, mock_kafka_service = mock_dependencies
    mock_kafka_service.list_all_topics.return_value = ["topic1", "topic2"]

    response = client.get("/kafka/topics")

    assert response.status_code == 200
    assert response.json() == ["topic1", "topic2"]


def test_get_all_topics_failure(mock_dependencies):
    client, mock_kafka_service = mock_dependencies
    mock_kafka_service.list_all_topics.side_effect = Exception("Error")

    response = client.get("/kafka/topics")

    assert response.status_code == 500


def test_send_message_success(mock_dependencies):
    client, mock_kafka_service = mock_dependencies

    response = client.post("/kafka/topic/test_topic", json={"key": "k1", "value": {"data": 1}})

    assert response.status_code == 200
    mock_kafka_service.send_message.assert_called_once_with("test_topic", "k1", '{"data": 1}')


def test_consume_message_success(mock_dependencies):
    client, mock_kafka_service = mock_dependencies
    mock_kafka_service.consume_topic.return_value = ["msg1"]

    response = client.get("/kafka/topic/test_topic")

    assert response.status_code == 200
    assert response.json() == ["msg1"]
    mock_kafka_service.consume_topic.assert_called_once_with("test_topic", 10)


def test_create_topic_success(mock_dependencies):
    client, mock_kafka_service = mock_dependencies

    response = client.post("/kafka/topic/test_topic/create")

    assert response.status_code == 200
    mock_kafka_service.create_topic.assert_called_once_with("test_topic")


def test_delete_topic_success(mock_dependencies):
    client, mock_kafka_service = mock_dependencies

    response = client.delete("/kafka/topic/test_topic/delete")

    assert response.status_code == 200
    mock_kafka_service.delete_topic.assert_called_once_with("test_topic")
