import json
import logging

import mlflow
import pandas as pd

from config.config_loader import ConfigLoader
from config.kafka_config import KafkaConfigLoader
from src.pipelines.feature_engineering.fraud_feature_engineering import FraudFeatureEngineering
from src.pipelines.preprocessors.fraud_preprocessor import FraudPreprocessor
from src.services.kafka_service import KafkaService

logger = logging.getLogger(__name__)

class FraudService:
    def __init__(self, config_loader: ConfigLoader, model_type: str, model_id: str):
        self.config_loader = config_loader
        mlflow.set_tracking_uri(self.config_loader.config["mlflow"]["url"])
        self.model_type = model_type
        self.bucket_path = config_loader.config["mlflow"]["bucket_path"]
        self.model = self._load_model_by_run_id(model_id)
        self.fraud_features = FraudFeatureEngineering(config_loader)
        self.preprocessor = FraudPreprocessor(config_loader)
        self.kafka_config_loader = KafkaConfigLoader(config_loader)
        self.kafka_service = KafkaService(self.kafka_config_loader)
        self.fraud_detection_config = config_loader.config["api"]["fraud_detection"]

    def _load_model_by_run_id(self, model_id: str):
        try:
            logger.info(f"Loading model with id={model_id}")
            model_uri = f"{self.bucket_path}/{model_id}/artifacts"

            if self.model_type == "xgboost":
                model = mlflow.xgboost.load_model(model_uri)
            elif self.model_type == "random_forest" or self.model_type == "decision_tree":
                model = mlflow.sklearn.load_model(model_uri)
            else:
                model = mlflow.pyfunc.load_model(model_uri)
            logger.info(f"Model id={model_id} load successfully")

            return model
        except Exception as e:
            logger.error(f"Failed to load model id={model_id}")
            raise e

    def predict_transaction(self, transaction: dict) -> dict:
        transaction_df = pd.DataFrame([transaction])
        transaction_id = transaction["transaction_id"]

        transaction_df_processed = self.fraud_features.process(transaction_df)
        transaction_df_cleaned = self.preprocessor.clean_features(transaction_df_processed)

        prediction = self.model.predict(transaction_df_cleaned)[0]

        try:
            probabilities = self.model.predict_proba(transaction_df_cleaned)
            fraud_probability = float(probabilities[0, 1])
        except AttributeError:
            fraud_probability = float(prediction)

        confidence = self._get_confidence_level(fraud_probability)

        result = {
            "transaction_id": transaction_id,
            "is_fraud": bool(prediction),
            "fraud_probability": fraud_probability,
            "confidence": confidence
        }

        return result

    def fraud_handler(self, msg_value):
        kafka_service = KafkaService(self.kafka_config_loader)

        transaction = json.loads(msg_value.decode("utf-8"))
        decision = self.predict_transaction(transaction)

        if decision["is_fraud"]:
            fraud_alerts_topic = self.fraud_detection_config["kafka"]["fraud_alerts_topic"]
            kafka_service.send_message(fraud_alerts_topic, decision["transaction_id"], str(decision))
        else:
            decision_topic = self.fraud_detection_config["kafka"]["decision_topic"]
            kafka_service.send_message(decision_topic, decision["transaction_id"], str(decision))

    def _get_confidence_level(self, probability: float) -> str:
        if probability >= 0.8 or probability <= 0.2:
            return "high"
        elif probability >= 0.6 or probability <= 0.4:
            return "medium"
        else:
            return "low"