import asyncio
import io
import logging
import time

import pandas as pd
from fastapi import APIRouter, Request
from starlette.responses import JSONResponse

from config.config_loader import ConfigLoader
from config.kafka_config import KafkaConfigLoader
from src.kafka_producers.transaction_producer import TransactionProducer

router = APIRouter(prefix="/transaction")

config_loader = ConfigLoader()

logger = logging.getLogger(__name__)


@router.get("/{time_interval}", response_model=dict)
async def generate_transaction(request: Request, time_interval: int):
    try:
        response = request.app.state.generator.generate_transaction(time_interval)
        return JSONResponse(status_code=200, content=response)
    except Exception as e:
        logger.error("Failed to generate transaction", exc_info=True)
        return JSONResponse(
            status_code=400,
            content={"message": f"Failed to generate transaction: {e}"}
        )

@router.get("/fraud/{time_interval}", response_model=dict)
async def generate_fraud_transaction(request: Request, time_interval: int):
    try:
        response = request.app.state.generator.generate_fraudulent_transaction(time_interval)
        return JSONResponse(status_code=200, content=response)
    except Exception as e:
        logger.error("Failed to generate fraudulent transaction", exc_info=True)
        return JSONResponse(
            status_code=400,
            content={"message": f"Failed to generate fraudulent transaction: {e}"}
        )

@router.get("/normal/{time_interval}", response_model=dict)
async def generate_normal_transaction(request: Request, time_interval: int):
    try:
        response = request.app.state.generator.generate_normal_transaction(time_interval)
        return JSONResponse(status_code=200, content=response)
    except Exception as e:
        logger.error("Failed to generate normal transaction", exc_info=True)
        return JSONResponse(
            status_code=400,
            content={"message": f"Failed to generate normal transaction: {e}"}
        )

@router.post("/inject", response_model=dict)
async def inject_transactions(request: Request, duration_seconds: int):
    kafka_config_loader = KafkaConfigLoader(config_loader)

    transaction_producer = TransactionProducer(
        topic=config_loader.config["fraud_generator"]["topic"],
        kafka_config_loader=kafka_config_loader,
        data_generator=request.app.state.generator.generate_transaction
    )

    transaction_producer.start_loading(duration_seconds=duration_seconds)
    await asyncio.sleep(duration_seconds)
    transaction_producer.stop_loading()

    return JSONResponse(status_code=200, content="Injection completed")
