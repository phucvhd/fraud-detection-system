import logging
import os

from dotenv import load_dotenv


def pipeline_env_setup():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    load_dotenv()

    mlflow_s3_endpoint = os.environ.get("MLFLOW_S3_ENDPOINT_URL")
    minio_key = os.environ.get("MINIO_ACCESS_KEY_ID")
    minio_secret = os.environ.get("MINIO_SECRET_ACCESS_KEY")

    os.environ["AWS_ACCESS_KEY_ID"] = minio_key
    os.environ["AWS_SECRET_ACCESS_KEY"] = minio_secret
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = mlflow_s3_endpoint