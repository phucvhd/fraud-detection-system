import logging
import os
from dotenv import load_dotenv

from config.config_loader import ConfigLoader
from src.pipelines.fraud_pipeline import FraudPipeline
from src.pipelines.setup import pipeline_env_setup

if __name__ == '__main__':
    config_loader = ConfigLoader()
    pipeline_env_setup()

    fraud_pipeline = FraudPipeline(config_loader=config_loader)
    fraud_pipeline.run_mlflow_experiment_v3()
