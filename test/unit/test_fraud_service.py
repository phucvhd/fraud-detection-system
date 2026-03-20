import json
from unittest.mock import Mock, patch
import pandas as pd
import pytest

from src.services.fraud_service import FraudService
from src.schemas.transaction import TransactionCanonical

@pytest.fixture
def mock_config_loader():
    config = Mock()
    config.config = {
        "api": {
            "fraud_detection": {
                "model": {"id": "test_model_1"},
                "kafka": {
                    "fraud_alerts_topic": "alerts",
                    "decision_topic": "decisions"
                }
            }
        },
        "preprocessor": {
            "features_to_scale": [f"V{i}" for i in range(1, 29)]
        }
    }
    return config

@pytest.fixture
@patch("src.services.fraud_service.S3Client")
@patch("src.services.fraud_service.KafkaConfigLoader")
@patch("src.services.fraud_service.KafkaService")
@patch("src.services.fraud_service.tarfile")
@patch("src.services.fraud_service.joblib")
def fraud_service(mock_joblib, mock_tarfile, mock_kafka_service, mock_kafka_config, mock_s3_client, mock_config_loader):
    mock_model = Mock()
    mock_model.predict.return_value = [1]
    mock_model.predict_proba.return_value = [[0.1, 0.9]]
    mock_joblib.load.return_value = mock_model

    mock_s3_instance = mock_s3_client.return_value
    mock_s3_instance.get_object.return_value = b"mocked tar file content"

    mock_member = Mock()
    mock_member.name = "model.joblib"
    mock_tar = Mock()
    mock_tar.getmembers.return_value = [mock_member]
    mock_tar.extractfile.return_value = Mock()
    mock_tarfile.open.return_value = mock_tar

    service = FraudService(mock_config_loader)
    return service

def test_predict_transaction(fraud_service):
    transaction = {
        "transaction_id": "123e4567-e89b-12d3-a456-426614174000",
        "Time": 3600,
        "Amount": 100.0,
    }
    for i in range(1, 29):
        transaction[f"V{i}"] = 0.5
        
    result = fraud_service.predict_transaction(transaction)
    
    assert isinstance(result, TransactionCanonical)
    assert str(result.transaction_id) == "123e4567-e89b-12d3-a456-426614174000"
    assert result.is_fraud is True
    assert result.amount == 100.0
    assert result.event_time_seconds == 3600

def test_fraud_handler(fraud_service):
    transaction = {
        "transaction_id": "123e4567-e89b-12d3-a456-426614174000",
        "Time": 3600,
        "Amount": 100.0,
    }
    for i in range(1, 29):
        transaction[f"V{i}"] = 0.5
        
    msg_value = json.dumps(transaction).encode("utf-8")
    
    with patch("src.services.fraud_service.KafkaService") as mock_kafka:
        kafka_instance = mock_kafka.return_value
        fraud_service.fraud_handler(msg_value)
        
        assert kafka_instance.send_message.call_count == 2
        calls = kafka_instance.send_message.call_args_list
        assert calls[0][0][0] == "alerts"
        assert calls[1][0][0] == "decisions"

def test_get_confidence_level(fraud_service):
    assert fraud_service._get_confidence_level(0.9) == "high"
    assert fraud_service._get_confidence_level(0.1) == "high"
    assert fraud_service._get_confidence_level(0.7) == "medium"
    assert fraud_service._get_confidence_level(0.5) == "low"

def test_get_hour_risk_score(fraud_service):
    assert fraud_service.get_hour_risk_score(2) == 0.003968
    assert fraud_service.get_hour_risk_score(6) == 0.005402

def test_add_time_features(fraud_service):
    df = pd.DataFrame([{"Time": 7200}])
    result = fraud_service.add_time_features(df)
    
    assert "hour_of_day" in result.columns
    assert result["hour_of_day"].iloc[0] == 2.0
    assert "day_period" in result.columns
    assert "time_since_start" in result.columns

def test_add_amount_features(fraud_service):
    df = pd.DataFrame([{"Amount": 100.0}])
    result = fraud_service.add_amount_features(df)
    
    assert "log_amount" in result.columns
    assert "amount_scaled" in result.columns

def test_process(fraud_service):
    df = pd.DataFrame([{"Time": 3600, "Amount": 50.0}])
    result = fraud_service.process(df)
    
    assert "hour_of_day" in result.columns
    assert "log_amount" in result.columns

def test_clean_features(fraud_service):
    data = {
        "Time": [3600],
        "Amount": [50.0],
        "hour_of_day": [1.0],
        "day_period": [0],
        "time_since_start": [1.0],
        "log_amount": [4.0],
        "amount_scaled": [0.0]
    }
    for i in range(1, 29):
        data[f"V{i}"] = [0.0]
        
    df = pd.DataFrame(data)
    result = fraud_service.clean_features(df)
    
    assert len(result.columns) == 35
    assert "Time" in result.columns
    assert "Amount" in result.columns
    assert "V1" in result.columns
