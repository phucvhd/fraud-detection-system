import asyncio
import logging

from fastapi import APIRouter, Request
from starlette.responses import JSONResponse

from src.kafka_producers.transaction_producer import TransactionProducer

router = APIRouter(prefix="/transaction")
logger = logging.getLogger(__name__)


@router.get("/fraud/{time_interval}")
async def generate_fraud_transaction(request: Request, time_interval: int):
    try:
        response = request.app.state.generator.generate_fraudulent_transaction(time_interval)
        return JSONResponse(status_code=200, content=response)
    except Exception:
        logger.error("Failed to generate fraudulent transaction", exc_info=True)
        return JSONResponse(status_code=500, content={"message": "Failed to generate fraudulent transaction"})


@router.get("/normal/{time_interval}")
async def generate_normal_transaction(request: Request, time_interval: int):
    try:
        response = request.app.state.generator.generate_normal_transaction(time_interval)
        return JSONResponse(status_code=200, content=response)
    except Exception:
        logger.error("Failed to generate normal transaction", exc_info=True)
        return JSONResponse(status_code=500, content={"message": "Failed to generate normal transaction"})


@router.get("/{time_interval}")
async def generate_transaction(request: Request, time_interval: int):
    try:
        response = request.app.state.generator.generate_transaction(time_interval)
        return JSONResponse(status_code=200, content=response)
    except Exception:
        logger.error("Failed to generate transaction", exc_info=True)
        return JSONResponse(status_code=500, content={"message": "Failed to generate transaction"})


@router.post("/inject")
async def inject_transactions(request: Request, duration_seconds: int):
    config_loader = request.app.state.config_loader
    producer = TransactionProducer(
        topic=config_loader.config["fraud_generator"]["topic"],
        kafka_config_loader=request.app.state.kafka_config_loader,
        data_generator=request.app.state.generator.generate_transaction,
    )
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, lambda: producer.start_loading(duration_seconds=duration_seconds))
    producer.stop_loading()
    return JSONResponse(status_code=200, content="Injection completed")
