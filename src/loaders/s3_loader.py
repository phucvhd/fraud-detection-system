import json
import logging
import os
from datetime import datetime

import boto3
from botocore.exceptions import ClientError

from config.config_loader import ConfigLoader

logger = logging.getLogger(__name__)


class S3Client:
    def __init__(self, config_loader: ConfigLoader):
        self.AWS_ACCESS_KEY_ID = config_loader.config["aws"]["credentials"]["access_key_id"]
        self.AWS_SECRET_ACCESS_KEY = config_loader.config["aws"]["credentials"]["secret_access_key"]
        self.AWS_DEFAULT_REGION = config_loader.config["aws"]["region"]
        self.AWS_BUCKET_NAME = config_loader.config["aws"]["s3"]["bucket_name"]

        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=self.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=self.AWS_SECRET_ACCESS_KEY,
            region_name=self.AWS_DEFAULT_REGION
        )

    def upload_json(self, data: dict, s3_key: str) -> None:
        try:
            json_string = json.dumps(data, indent=2, ensure_ascii=False)

            self.s3_client.put_object(
                Bucket=self.AWS_BUCKET_NAME,
                Key=s3_key,
                Body=json_string,
                ContentType='application/json'
            )

            logger.info(f"✓ Uploaded: s3://{self.AWS_BUCKET_NAME}/{s3_key}")
        except Exception as e:
            logger.error(f"✗ Upload failed for {s3_key}: {e}")
            raise

    def upload_file(self, local_path: str, s3_key: str) -> None:
        try:
            self.s3_client.upload_file(local_path, self.AWS_BUCKET_NAME, s3_key)
            logger.info(f"✓ Uploaded: {local_path} → s3://{self.AWS_BUCKET_NAME}/{s3_key}")
        except Exception as e:
            logger.error(f"✗ Upload failed: {e}")
            raise

    def get_object(self, s3_key: str):
        try:
            response = self.s3_client.get_object(
                Bucket=self.AWS_BUCKET_NAME,
                Key=s3_key
            )

            content = response['Body'].read()

            logger.info(f"✓ Retrieved: s3://{self.AWS_BUCKET_NAME}/{s3_key}")
            return content
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                logger.error(f"✗ File not found: {s3_key}")
            else:
                logger.error(f"✗ AWS error: {e}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"✗ Invalid JSON: {e}")
            raise

    def download_file(self, s3_key: str, local_path: str) -> None:
        try:
            self.s3_client.download_file(self.AWS_BUCKET_NAME, s3_key, local_path)
            logger.info(f"✓ Downloaded: s3://{self.AWS_BUCKET_NAME}/{s3_key} → {local_path}")
        except Exception as e:
            logger.error(f"✗ Download failed: {e}")
            raise

    def list_objects(self, prefix: str = '') -> list[str]:
        try:
            keys = []
            continuation_token = None

            while True:
                params = {
                    'Bucket': self.AWS_BUCKET_NAME,
                    'Prefix': prefix,
                    'MaxKeys': 1000
                }

                if continuation_token:
                    params['ContinuationToken'] = continuation_token

                response = self.s3_client.list_objects_v2(**params)

                if 'Contents' not in response:
                    break

                for obj in response['Contents']:
                    keys.append(obj['Key'])

                if not response.get('IsTruncated', False):
                    break

                continuation_token = response.get('NextContinuationToken')

            logger.info(f"✓ Found {len(keys)} objects in '{prefix}'")
            return keys

        except Exception as e:
            logger.error(f"✗ List failed: {e}")
            return []

    def get_all_jsons_from_folder(self, prefix: str) -> list[dict]:
        try:
            keys = self.list_objects(prefix)

            # Filter only JSON files
            json_keys = [k for k in keys if k.endswith('.json')]

            results = []
            for key in json_keys:
                data = self.get_object(key)
                results.append({
                    'key': key,
                    'data': data
                })

            logger.info(f"✓ Retrieved {len(results)} JSON files from '{prefix}'")
            return results

        except Exception as e:
            logger.error(f"✗ Get all JSONs failed: {e}")
            return []