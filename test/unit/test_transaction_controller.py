from unittest.mock import Mock, patch
from fastapi.testclient import TestClient
from fastapi import FastAPI
import pytest

@pytest.fixture(autouse=True)
def mock_dependencies():
    with patch("src.controllers.transaction_controller.ConfigLoader"), \
         patch("src.controllers.transaction_controller.S3Client") as mock_s3_client, \
         patch("src.controllers.transaction_controller.pd.read_csv"), \
         patch("src.controllers.transaction_controller.FraudSyntheticDataGenerator") as mock_generator, \
         patch("src.controllers.transaction_controller.TransactionProducer") as mock_producer, \
         patch("src.controllers.transaction_controller.time.sleep"):

        # Mock ConfigLoader config
        from src.controllers import transaction_controller
        transaction_controller.config_loader.config = {
            "data": {"raw": {"s3": "test_s3_key"}},
            "fraud_generator": {"topic": "test_topic"}
        }
        # Reset lazy generator so each test starts fresh
        transaction_controller._generator = None

        from src.controllers.transaction_controller import router

        app = FastAPI()
        app.include_router(router)
        client = TestClient(app)

        yield client, mock_generator, mock_producer

def test_generate_transaction_success(mock_dependencies):
    client, mock_generator, _ = mock_dependencies
    mock_generator.return_value.generate_transaction.return_value = {"id": "tx1", "Time": 100}

    response = client.get("/transaction/100")

    assert response.status_code == 200
    assert response.json() == {"id": "tx1", "Time": 100}
    mock_generator.return_value.generate_transaction.assert_called_once_with(100)

def test_generate_fraud_transaction_success(mock_dependencies):
    client, mock_generator, _ = mock_dependencies
    mock_generator.return_value.generate_fraudulent_transaction.return_value = {"id": "tx2", "Time": 200, "Class": 1}

    response = client.get("/transaction/fraud/200")

    assert response.status_code == 200
    assert response.json() == {"id": "tx2", "Time": 200, "Class": 1}
    mock_generator.return_value.generate_fraudulent_transaction.assert_called_once_with(200)

def test_generate_normal_transaction_success(mock_dependencies):
    client, mock_generator, _ = mock_dependencies
    mock_generator.return_value.generate_normal_transaction.return_value = {"id": "tx3", "Time": 300, "Class": 0}

    response = client.get("/transaction/normal/300")

    assert response.status_code == 200
    assert response.json() == {"id": "tx3", "Time": 300, "Class": 0}
    mock_generator.return_value.generate_normal_transaction.assert_called_once_with(300)

def test_inject_transactions(mock_dependencies):
    client, _, mock_producer = mock_dependencies

    response = client.post("/transaction/inject?duration_seconds=5")

    assert response.status_code == 200
    mock_producer.return_value.start_loading.assert_called_once_with(duration_seconds=5)
    mock_producer.return_value.stop_loading.assert_called_once()
