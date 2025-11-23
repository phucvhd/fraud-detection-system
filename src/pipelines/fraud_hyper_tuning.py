import logging
import os

from dotenv import load_dotenv

from config.config_loader import ConfigLoader
from src.pipelines.fraud_pipeline import FraudPipeline

logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
config_loader = ConfigLoader()

load_dotenv()
s3_endpoint = os.environ.get("MLFLOW_S3_ENDPOINT_URL")
aws_key = os.environ.get("AWS_ACCESS_KEY_ID")
aws_secret = os.environ.get("AWS_SECRET_ACCESS_KEY")

os.environ["AWS_ACCESS_KEY_ID"] = aws_key
os.environ["AWS_SECRET_ACCESS_KEY"] = aws_secret
os.environ["MLFLOW_S3_ENDPOINT_URL"] = s3_endpoint

if __name__ == '__main__':
    fraud_pipeline = FraudPipeline(config_loader=config_loader)
    fraud_pipeline.run_mlflow_hyper_tune()
