import json
import logging
from fastapi import APIRouter
from starlette.responses import JSONResponse

from config.config_loader import ConfigLoader
from config.kafka_config import KafkaConfigLoader
from src.services.fraud_service import FraudService
from src.services.kafka_service import KafkaService

router = APIRouter(prefix="/fraud")

config_loader = ConfigLoader()
kafka_config_loader = KafkaConfigLoader(config_loader)

fraud_detection_config = config_loader.config["api"]["fraud_detection"]
fraud_service = FraudService(config_loader)

logger = logging.getLogger(__name__)

@router.post("/validate", response_model=dict)
def validate_fraud(request: dict):
    try:
        decision = fraud_service.predict_transaction(request)
        kafka_service = KafkaService(kafka_config_loader)

        if decision["is_fraud"]:
            fraud_alerts_topic = fraud_detection_config["kafka"]["fraud_alerts_topic"]
            kafka_service.send_message(fraud_alerts_topic, decision["transaction_id"], str(decision))
            logger.info(f"An alert for transaction={decision['transaction_id']} has been sent to {fraud_alerts_topic}")

        decision_topic = fraud_detection_config["kafka"]["decision_topic"]
        kafka_service.send_message(decision_topic, decision["transaction_id"], str(decision))
        logger.info(f"Decision for transaction={decision['transaction_id']} has been sent to {decision_topic}")

        return JSONResponse(
            status_code=200,
            content=decision
        )
    except Exception as e:
        logger.error(f"Failed to validate transaction_id: {request['transaction_id']}", e)
        return JSONResponse(
            status_code=400,
            content={f"message: Failed to validate transaction"}
        )
