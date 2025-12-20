import logging
from fastapi import APIRouter
from starlette.responses import JSONResponse

from config.config_loader import ConfigLoader
from src.pipelines.feature_engineering.fraud_feature_engineering import FraudFeatureEngineering
from src.services.fraud_service import FraudService

router = APIRouter(prefix="/fraud")

config_loader = ConfigLoader()
fraud_detection_config = config_loader.config["api"]["fraud_detection"]
model_type = fraud_detection_config["model"]["type"]
model_id = fraud_detection_config["model"]["id"]
mlflow_service = FraudService(config_loader, model_type, model_id)

logger = logging.getLogger(__name__)

@router.post("/validate", response_model=dict)
def validate_fraud(request: dict):
    try:
        response = mlflow_service.predict_transaction(request)
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
