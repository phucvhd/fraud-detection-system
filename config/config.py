import os
from pathlib import Path

from dotenv import load_dotenv


class Config:
    def __init__(self, profile_env=None):
        profile = os.getenv("ENV_PROFILE", profile_env)
        dotenv_file = f".env.{profile}" if profile_env else ".env"
        load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / dotenv_file)
        self.transactions_per_second = int(os.getenv("KAFKA_STREAM_TRANSACTION_PER_SECOND", 100))
        self.burst_mode = os.getenv("KAFKA_STREAM_BURST_MODE", "False").lower() == "true"
        self.burst_interval_seconds = int(os.getenv("KAFKA_STREAM_BURST_INTERVAL_SECONDS", 30))
        self.burst_multiplier = float(os.getenv("KAFKA_STREAM_BURST_MULTIPLIER", 1.0))
        self.bootstrap_servers = os.getenv("KAFKA_STREAM_BOOTSTRAP_SERVERS", "localhost:9092").strip('"')
        self.topic = os.getenv("KAFKA_STREAM_TOPIC", "transactions").strip('"')
        self.fraud_rate = float(os.getenv("KAFKA_STREAM_FRAUD_RATE", 0.005))
        self.max_transaction_time=int(os.getenv("FRAUD_GENERATOR_MAX_TRANSACTION_TIME", 172800))