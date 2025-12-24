import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI

from config.config_loader import ConfigLoader
from config.kafka_config import KafkaConfigLoader
from src.controllers.fraud_detection_controller import router as fraud_router
from src.controllers.kafka_controller import router as kafka_router
from src.controllers.transaction_controller import router as transaction_router
from src.kafka_consumers.fraud_listener import FraudListener
from src.services.fraud_service import FraudService

config_loader = ConfigLoader()
kafka_config_loader = KafkaConfigLoader(config_loader)

fraud_detection_config = config_loader.config["api"]["fraud_detection"]
model_type = fraud_detection_config["model"]["type"]
model_id = fraud_detection_config["model"]["id"]
fraud_service = FraudService(config_loader, model_type, model_id)
fraud_listener = FraudListener(config_loader)

@asynccontextmanager
async def lifespan(app: FastAPI):
    thread = threading.Thread(
        target=fraud_listener.start_listener,
        args=(fraud_service.fraud_handler,),
        daemon=True)
    thread.start()
    yield

app = FastAPI(lifespan=lifespan)

app.include_router(fraud_router)
app.include_router(kafka_router)
app.include_router(transaction_router)