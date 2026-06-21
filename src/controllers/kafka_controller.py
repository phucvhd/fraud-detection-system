import json
import logging

from fastapi import APIRouter, Request
from starlette.responses import JSONResponse

router = APIRouter(prefix="/kafka")
logger = logging.getLogger(__name__)


@router.get("/topics")
async def get_all_topics(request: Request):
    try:
        topics = request.app.state.kafka_service.list_all_topics()
        return JSONResponse(status_code=200, content=topics)
    except Exception:
        logger.error("Failed to list Kafka topics", exc_info=True)
        return JSONResponse(status_code=500, content={"message": "Failed to list Kafka topics"})


@router.post("/topic/{topic_name}")
async def send_message(request: Request, topic_name: str, message: dict):
    try:
        message_str = json.dumps(message["value"])
        request.app.state.kafka_service.send_message(topic_name, message["key"], message_str)
        return JSONResponse(status_code=200, content=f"Message sent to topic={topic_name}")
    except Exception:
        logger.error("Failed to send message to topic=%s", topic_name, exc_info=True)
        return JSONResponse(status_code=500, content={"message": f"Failed to send message to topic={topic_name}"})


@router.get("/topic/{topic_name}")
async def consume_message(request: Request, topic_name: str):
    try:
        messages = request.app.state.kafka_service.consume_topic(topic_name, 10)
        return JSONResponse(status_code=200, content=messages)
    except Exception:
        logger.error("Failed to consume from topic=%s", topic_name, exc_info=True)
        return JSONResponse(status_code=500, content={"message": f"Failed to consume from topic={topic_name}"})


@router.post("/topic/{topic_name}/create")
async def create_topic(request: Request, topic_name: str):
    try:
        request.app.state.kafka_service.create_topic(topic_name)
        return JSONResponse(status_code=200, content=f"Topic={topic_name} created")
    except Exception:
        logger.error("Failed to create topic=%s", topic_name, exc_info=True)
        return JSONResponse(status_code=500, content={"message": f"Failed to create topic={topic_name}"})


@router.delete("/topic/{topic_name}/delete")
async def delete_topic(request: Request, topic_name: str):
    try:
        request.app.state.kafka_service.delete_topic(topic_name)
        return JSONResponse(status_code=200, content=f"Topic={topic_name} deleted")
    except Exception:
        logger.error("Failed to delete topic=%s", topic_name, exc_info=True)
        return JSONResponse(status_code=500, content={"message": f"Failed to delete topic={topic_name}"})
