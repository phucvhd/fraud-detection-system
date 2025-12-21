import json
import logging

from fastapi import APIRouter
from starlette.responses import JSONResponse

from config.config_loader import ConfigLoader
from config.kafka_config import KafkaConfigLoader
from src.services.kafka_service import KafkaService

router = APIRouter(prefix="/kafka")

config_loader = ConfigLoader()
kafka_config = KafkaConfigLoader(config_loader)
kafka_service = KafkaService(kafka_config)

logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

@router.get("/topics", response_model=dict)
def get_all_topics():
    try:
        response = kafka_service.list_all_topics()
        return JSONResponse(
                status_code=200,
                content=response
            )
    except Exception:
        return JSONResponse(
            status_code=400,
            content={f"message: Failed to get topics from kafka"}
        )

@router.post("/topic/{topic_name}")
def send_message(topic_name: str, message: dict):
    try:
        message_str = json.dumps(message["value"])
        kafka_service.send_message(topic_name, message["key"], message_str)
        return JSONResponse(
                status_code=200,
                content=f"Message sent to topic={topic_name} successfully"
            )
    except Exception:
        return JSONResponse(
            status_code=400,
            content={f"message: Failed to send message to topic={topic_name}"}
        )

@router.get("/topic/{topic_name}")
def consume_message(topic_name: str):
    try:
        response = kafka_service.consume_topic(topic_name, 10)
        return JSONResponse(
                status_code=200,
                content=response
            )
    except Exception:
        return JSONResponse(
            status_code=400,
            content={f"message: Failed to consume message to topic={topic_name}"}
        )

@router.post("/topic/{topic_name}/create")
def create_topic(topic_name: str):
    try:
        kafka_service.create_topic(topic_name)
        return JSONResponse(
                status_code=200,
                content=f"Topic={topic_name} created successfully"
            )
    except Exception:
        return JSONResponse(
            status_code=400,
            content={f"message: Failed to create topic={topic_name}"}
        )

@router.delete("/topic/{topic_name}/delete")
def delete_topic(topic_name: str):
    try:
        kafka_service.delete_topic(topic_name)
        return JSONResponse(
                status_code=200,
                content=f"Topic={topic_name} deleted successfully"
            )
    except Exception:
        return JSONResponse(
            status_code=400,
            content={f"message: Failed to delete topic={topic_name}"}
        )