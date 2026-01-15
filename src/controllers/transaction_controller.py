import io
import time

import pandas as pd
from fastapi import APIRouter
from starlette.responses import JSONResponse

from config.config_loader import ConfigLoader
from config.kafka_config import KafkaConfigLoader
from src.generators.fraud_synthetic_generator import FraudSyntheticDataGenerator
from src.kafka_producers.transaction_producer import TransactionProducer
from src.clients.s3_client import S3Client

router = APIRouter(prefix="/transaction")

config_loader = ConfigLoader()
s3_client = S3Client(config_loader)
s3_key = config_loader.config["data"]["raw"]["s3"]
obj = s3_client.get_object(s3_key)
source_data = pd.read_csv(io.BytesIO(obj))
generator = FraudSyntheticDataGenerator(config_loader, source_data)

@router.get("/{time_interval}", response_model=dict)
def generate_transaction(time_interval: int):
    try:
        response = generator.generate_transaction(time_interval)
        return JSONResponse(
            status_code=200,
            content=response
        )
    except Exception:
        return JSONResponse(
            status_code=400,
            content={f"message: Failed to generate transaction"}
        )

@router.get("/fraud/{time_interval}", response_model=dict)
def generate_fraud_transaction(time_interval: int):
    try:
        response = generator.generate_fraudulent_transaction(time_interval)
        return JSONResponse(
            status_code=200,
            content=response
        )
    except Exception:
        return JSONResponse(
            status_code=400,
            content={f"message: Failed to generate transaction"}
        )

@router.get("/normal/{time_interval}", response_model=dict)
def generate_fraud_transaction(time_interval: int):
    try:
        response = generator.generate_normal_transaction(time_interval)
        return JSONResponse(
            status_code=200,
            content=response
        )
    except Exception:
        return JSONResponse(
            status_code=400,
            content={f"message: Failed to generate transaction"}
        )

@router.post("/inject", response_model=dict)
def inject_transactions(duration_seconds: int):
    kafka_config_loader = KafkaConfigLoader(config_loader)

    transaction_producer = TransactionProducer(
        topic=config_loader.config["fraud_generator"]["topic"],
        kafka_config_loader=kafka_config_loader,
        data_generator=generator.generate_transaction
    )

    transaction_producer.start_loading(duration_seconds=duration_seconds)
    time.sleep(duration_seconds)
    transaction_producer.stop_loading()

    return JSONResponse(
        status_code=200,
        content="Injection completed"
    )