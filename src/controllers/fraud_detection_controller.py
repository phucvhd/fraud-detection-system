import json
import logging
from fastapi import APIRouter
from starlette.responses import JSONResponse

from config.config_loader import ConfigLoader
from config.kafka_config import KafkaConfigLoader
from src.kafka_consumers.kafka_listener import KafkaListener
from src.services.fraud_service import FraudService
from src.services.kafka_service import KafkaService

router = APIRouter(prefix="/fraud")

config_loader = ConfigLoader()
kafka_config_loader = KafkaConfigLoader(config_loader)

fraud_detection_config = config_loader.config["api"]["fraud_detection"]
model_type = fraud_detection_config["model"]["type"]
model_id = fraud_detection_config["model"]["id"]
fraud_service = FraudService(config_loader, model_type, model_id)

input_topic = fraud_detection_config["kafka"]["topic"]
decision_topic = fraud_detection_config["kafka"]["decision_topic"]

logger = logging.getLogger(__name__)

@router.post("/validate", response_model=dict)
def validate_fraud(request: dict):
    try:
        response = fraud_service.predict_transaction(request)
        return JSONResponse(
            status_code=200,
            content=response
        )
    except Exception as e:
        logger.error(f"Failed to validate transaction_id: {request["transaction_id"]}", e)
        return JSONResponse(
            status_code=400,
            content={f"message: Failed to validate transaction"}
        )

kafka_service = KafkaService(kafka_config_loader)

def fraud_handler(msg_value):
    transaction = json.loads(msg_value.decode("utf-8"))
    decision = fraud_service.predict_transaction(transaction)
    kafka_service.send_message(decision_topic, decision["transaction_id"], str(decision))

kafka_listener = KafkaListener(input_topic, fraud_handler, kafka_config_loader)

@router.post("/validate/start", response_model=dict)
def start_fraud_validation():
    try:
        kafka_listener.start()
        return JSONResponse(
            status_code=200,
            content="message: Fraud listener started successfully"
        )
    except Exception as e:
        logger.error(f"Failed to start fraud listener", e)
        return JSONResponse(
            status_code=400,
            content="message: Failed to start fraud listener"
        )

@router.post("/validate/stop", response_model=dict)
def stop_fraud_validation():
    try:
        kafka_listener.stop()
        return JSONResponse(
            status_code=200,
            content="message: Fraud listener stopped successfully"
        )
    except Exception as e:
        logger.error(f"Failed to start fraud listener", e)
        return JSONResponse(
            status_code=400,
            content="message: Failed to stop fraud listener"
        )
