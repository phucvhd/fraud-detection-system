import logging

from fastapi import APIRouter, Request
from starlette.responses import JSONResponse

from src.schemas.transaction import TransactionCanonical

router = APIRouter(prefix="/fraud")
logger = logging.getLogger(__name__)


@router.post("/validate", response_model=TransactionCanonical)
async def validate_fraud(request: Request, body: dict):
    try:
        fraud_service = request.app.state.fraud_service
        kafka_service = request.app.state.kafka_service
        kafka_cfg = request.app.state.config_loader.config["api"]["fraud_detection"]["kafka"]

        decision = fraud_service.predict_transaction(body)

        if decision.is_fraud:
            kafka_service.send_message(
                kafka_cfg["fraud_alerts_topic"],
                str(decision.transaction_id),
                decision.model_dump(),
            )
            logger.info("Fraud alert sent for transaction=%s", decision.transaction_id)

        kafka_service.send_message(
            kafka_cfg["decision_topic"],
            str(decision.transaction_id),
            decision.model_dump(),
        )
        logger.info("Decision sent for transaction=%s", decision.transaction_id)

        return JSONResponse(status_code=200, content=decision.model_dump(mode="json"))
    except Exception:
        logger.error("Failed to validate transaction_id=%s", body.get("transaction_id"), exc_info=True)
        return JSONResponse(status_code=500, content={"message": "Failed to validate transaction"})
