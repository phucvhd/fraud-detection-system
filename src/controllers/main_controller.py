import io
import threading
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI

from config.config_loader import ConfigLoader
from config.kafka_config import KafkaConfigLoader
from src.clients.s3_client import S3Client
from src.controllers.fraud_detection_controller import router as fraud_router
from src.controllers.kafka_controller import router as kafka_router
from src.controllers.transaction_controller import router as transaction_router
from src.generators.fraud_synthetic_generator import FraudSyntheticDataGenerator
from src.kafka_consumers.fraud_listener import FraudListener
from src.services.fraud_service import FraudService

config_loader = ConfigLoader()
kafka_config_loader = KafkaConfigLoader(config_loader)

fraud_service = FraudService(config_loader)
fraud_listener = FraudListener(config_loader)

@asynccontextmanager
async def lifespan(app: FastAPI):
    s3_client = S3Client(config_loader)
    s3_key = config_loader.config["data"]["raw"]["s3"]
    obj = s3_client.get_object(s3_key)
    source_data = pd.read_csv(io.BytesIO(obj))
    app.state.generator = FraudSyntheticDataGenerator(config_loader, source_data)

    thread = threading.Thread(
        target=fraud_listener.start_listener,
        args=(fraud_service.fraud_handler,),
        daemon=True)
    thread.start()
    yield
    fraud_listener.stop_listener()

app = FastAPI(lifespan=lifespan)

@app.get("/health")
async def health():
    return {"status": "ok"}

app.include_router(fraud_router)
app.include_router(kafka_router)
app.include_router(transaction_router)
