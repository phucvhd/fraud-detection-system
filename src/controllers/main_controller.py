from fastapi import FastAPI
from src.controllers.fraud_detection_controller import router as fraud_router
from src.controllers.kafka_controller import router as kafka_router
from src.controllers.transaction_controller import router as transaction_router

app = FastAPI()

app.include_router(fraud_router)
app.include_router(kafka_router)
app.include_router(transaction_router)