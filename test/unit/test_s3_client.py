import json
import pytest
from botocore.exceptions import ClientError
from unittest.mock import Mock, patch

from src.clients.s3_client import S3Client

@pytest.fixture
def mock_config_loader():
    loader = Mock()
    loader.config = {
        "aws": {
            "credentials": {
                "access_key_id": "test_key",
                "secret_access_key": "test_secret"
            },
            "region": "us-east-1",
            "s3": {
                "bucket_name": "test-bucket"
            }
        }
    }
    return loader

@pytest.fixture
@patch("src.clients.s3_client.boto3.client")
def s3_client(mock_boto3, mock_config_loader):
    client = S3Client(mock_config_loader)
    client.s3_client = Mock()
    return client

def test_upload_json(s3_client):
    data = {"key": "value"}
    s3_key = "test_key.json"
    
    s3_client.upload_json(data, s3_key)
    
    s3_client.s3_client.put_object.assert_called_once_with(
        Bucket="test-bucket",
        Key=s3_key,
        Body=json.dumps(data, indent=2, ensure_ascii=False),
        ContentType="application/json"
    )

def test_upload_file(s3_client):
    local_path = "local.txt"
    s3_key = "remote.txt"
    
    s3_client.upload_file(local_path, s3_key)
    
    s3_client.s3_client.upload_file.assert_called_once_with(
        local_path, "test-bucket", s3_key
    )

def test_get_object_success(s3_client):
    s3_key = "data.json"
    mock_body = Mock()
    mock_body.read.return_value = b'{"hello": "world"}'
    s3_client.s3_client.get_object.return_value = {"Body": mock_body}
    
    result = s3_client.get_object(s3_key)
    
    assert result == b'{"hello": "world"}'
    s3_client.s3_client.get_object.assert_called_once_with(
        Bucket="test-bucket",
        Key=s3_key
    )

def test_get_object_not_found(s3_client):
    s3_key = "missing.json"
    error_response = {'Error': {'Code': 'NoSuchKey'}}
    s3_client.s3_client.get_object.side_effect = ClientError(error_response, 'GetObject')
    
    with pytest.raises(ClientError):
        s3_client.get_object(s3_key)

def test_download_file(s3_client):
    s3_key = "remote.txt"
    local_path = "local.txt"
    
    s3_client.download_file(s3_key, local_path)
    
    s3_client.s3_client.download_file.assert_called_once_with(
        "test-bucket", s3_key, local_path
    )

def test_list_objects(s3_client):
    prefix = "test/"
    s3_client.s3_client.list_objects_v2.side_effect = [
        {
            "Contents": [{"Key": "test/1.json"}, {"Key": "test/2.json"}],
            "IsTruncated": True,
            "NextContinuationToken": "token1"
        },
        {
            "Contents": [{"Key": "test/3.json"}],
            "IsTruncated": False
        }
    ]
    
    results = s3_client.list_objects(prefix)
    
    assert results == ["test/1.json", "test/2.json", "test/3.json"]
    assert s3_client.s3_client.list_objects_v2.call_count == 2

def test_get_all_jsons_from_folder(s3_client):
    prefix = "data/"
    s3_client.list_objects = Mock(return_value=["data/1.json", "data/2.txt", "data/3.json"])
    s3_client.get_object = Mock(side_effect=[b'{"a": 1}', b'{"b": 2}'])
    
    results = s3_client.get_all_jsons_from_folder(prefix)
    
    assert len(results) == 2
    assert results[0]["key"] == "data/1.json"
    assert results[0]["data"] == b'{"a": 1}'
    assert results[1]["key"] == "data/3.json"
    assert results[1]["data"] == b'{"b": 2}'
