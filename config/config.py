import os
from pathlib import Path

from dotenv import load_dotenv


def _load_env(profile_env):
    profile = os.getenv("ENV_PROFILE", profile_env)
    dotenv_file = f".env.{profile}" if profile_env else ".env"
    load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / dotenv_file)

class Config:
    def __init__(self, profile_env=None):
        _load_env(profile_env)
        self.fraud_rate = float(os.getenv("FRAUD_GENERATOR_FRAUD_RATE", 0.005))
        self.max_transaction_time=int(os.getenv("FRAUD_GENERATOR_MAX_TRANSACTION_TIME", 172800))
