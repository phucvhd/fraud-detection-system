from fastapi import FastAPI
from src.controllers.fraud_detection_controller import router as fraud_router

app = FastAPI()

app.include_router(fraud_router)