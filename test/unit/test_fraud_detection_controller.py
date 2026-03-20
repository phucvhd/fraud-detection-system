from unittest.mock import Mock, patch
from fastapi.testclient import TestClient
import pytest

@pytest.fixture(autouse=True)
def mock_dependencies():
    with patch("src.controllers.fraud_detection_controller.ConfigLoader"), \
         patch("src.controllers.fraud_detection_controller.KafkaConfigLoader"), \
         patch("src.controllers.fraud_detection_controller.FraudService") as mock_fraud_service, \
         patch("src.controllers.fraud_detection_controller.KafkaService") as mock_kafka_service, \
         patch("src.controllers.fraud_detection_controller.fraud_detection_config", {"kafka": {"fraud_alerts_topic": "alerts", "decision_topic": "decisions"}}):
             
        from src.controllers.fraud_detection_controller import router
        
        mock_decision = Mock()
        mock_decision.is_fraud = True
        mock_decision.transaction_id = "tx123"
        mock_decision.model_dump.return_value = {"id": "tx123", "is_fraud": True}
        
        mock_fraud_service.return_value.predict_transaction.return_value = mock_decision
        
        from fastapi import FastAPI
        app = FastAPI()
        app.include_router(router)
        client = TestClient(app)
        
        yield client, mock_fraud_service, mock_kafka_service

def test_validate_fraud_success(mock_dependencies):
    client, mock_fraud_service, mock_kafka_service = mock_dependencies
    
    response = client.post("/fraud/validate", json={"transaction_id": "tx123", "amount": 100})
    
    assert response.status_code == 200
    
    mock_fraud_service.return_value.predict_transaction.assert_called_once_with({"transaction_id": "tx123", "amount": 100})
    assert mock_kafka_service.return_value.send_message.call_count == 2
    
    calls = mock_kafka_service.return_value.send_message.call_args_list
    assert calls[0][0][0] == "alerts"
    assert calls[1][0][0] == "decisions"

def test_validate_fraud_failure(mock_dependencies):
    client, mock_fraud_service, mock_kafka_service = mock_dependencies
    
    mock_fraud_service.return_value.predict_transaction.side_effect = Exception("Test error")
    
    response = client.post("/fraud/validate", json={"transaction_id": "tx123"})
    
    assert response.status_code == 400
    assert response.json() == {"message": "Failed to validate transaction: Test error"}

