import json
import logging

import boto3
from botocore.exceptions import ClientError

from config.config_loader import ConfigLoader

logger = logging.getLogger(__name__)


class S3Client:
    def __init__(self, config_loader: ConfigLoader):
        aws = config_loader.config["aws"]
        self.bucket = config_loader.config["aws"]["s3"]["bucket_name"]
        self.s3_client = boto3.client(
            "s3",
            aws_access_key_id=aws["credentials"]["access_key_id"],
            aws_secret_access_key=aws["credentials"]["secret_access_key"],
            region_name=aws["region"],
        )

    def upload_json(self, data: dict, s3_key: str) -> None:
        try:
            self.s3_client.put_object(
                Bucket=self.bucket,
                Key=s3_key,
                Body=json.dumps(data, indent=2, ensure_ascii=False),
                ContentType="application/json",
            )
            logger.info("Uploaded JSON to s3://%s/%s", self.bucket, s3_key)
        except Exception:
            logger.error("Failed to upload JSON to s3://%s/%s", self.bucket, s3_key, exc_info=True)
            raise

    def upload_file(self, local_path: str, s3_key: str) -> None:
        try:
            self.s3_client.upload_file(local_path, self.bucket, s3_key)
            logger.info("Uploaded %s to s3://%s/%s", local_path, self.bucket, s3_key)
        except Exception:
            logger.error("Failed to upload %s", local_path, exc_info=True)
            raise

    def get_object(self, s3_key: str) -> bytes:
        try:
            response = self.s3_client.get_object(Bucket=self.bucket, Key=s3_key)
            content = response["Body"].read()
            logger.info("Retrieved s3://%s/%s", self.bucket, s3_key)
            return content
        except ClientError as e:
            code = e.response["Error"]["Code"]
            if code == "NoSuchKey":
                logger.error("Object not found: s3://%s/%s", self.bucket, s3_key)
            else:
                logger.error("AWS error fetching s3://%s/%s: %s", self.bucket, s3_key, e)
            raise

    def download_file(self, s3_key: str, local_path: str) -> None:
        try:
            self.s3_client.download_file(self.bucket, s3_key, local_path)
            logger.info("Downloaded s3://%s/%s to %s", self.bucket, s3_key, local_path)
        except Exception:
            logger.error("Failed to download s3://%s/%s", self.bucket, s3_key, exc_info=True)
            raise

    def list_objects(self, prefix: str = "") -> list[str]:
        try:
            keys = []
            paginator = self.s3_client.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
                for obj in page.get("Contents", []):
                    keys.append(obj["Key"])
            logger.info("Found %d objects under prefix='%s'", len(keys), prefix)
            return keys
        except Exception:
            logger.error("Failed to list objects under prefix='%s'", prefix, exc_info=True)
            return []

    def get_all_jsons_from_folder(self, prefix: str) -> list[dict]:
        try:
            json_keys = [k for k in self.list_objects(prefix) if k.endswith(".json")]
            results = [{"key": key, "data": self.get_object(key)} for key in json_keys]
            logger.info("Retrieved %d JSON files from prefix='%s'", len(results), prefix)
            return results
        except Exception:
            logger.error("Failed to retrieve JSONs from prefix='%s'", prefix, exc_info=True)
            return []
