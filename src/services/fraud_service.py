import io
import json
import logging

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from config.config_loader import ConfigLoader
from config.kafka_config import KafkaConfigLoader
from src.clients.s3_client import S3Client
from src.services.kafka_service import KafkaService

logger = logging.getLogger(__name__)

class FraudService:
    def __init__(self, config_loader: ConfigLoader):
        self.config_loader = config_loader
        model_id = config_loader.config["api"]["fraud_detection"]["model"]["id"]
        self.kafka_config_loader = KafkaConfigLoader(config_loader)
        self.kafka_service = KafkaService(self.kafka_config_loader)
        self.fraud_detection_config = config_loader.config["api"]["fraud_detection"]
        self.s3_client = S3Client(config_loader)
        self.model = self._load_model(model_id)
        self.fraud_amount_median = 9.25
        self.fraud_amount_mean = 122.21

    def _load_model(self, model_id: str):
        try:
            logger.info(f"Loading model with id={model_id}")
            obj = self.s3_client.get_object(f"{model_id}/artifacts/model.pkl")
            bytestream = io.BytesIO(obj)
            model = joblib.load(bytestream)
            logger.info(f"Model id={model_id} load successfully")
            return model
        except Exception as e:
            logger.error(f"Failed to load model id={model_id}")
            raise e

    def predict_transaction(self, transaction: dict) -> dict:
        transaction_df = pd.DataFrame([transaction])
        transaction_id = transaction["transaction_id"]

        transaction_df_processed = self.process(transaction_df)
        transaction_df_cleaned = self.clean_features(transaction_df_processed)

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

    def get_hour_risk_score(self, hour):
        # Based on your 4-hour period analysis
        if 0 <= hour < 4:
            return 0.003968
        elif 4 <= hour < 8:
            return 0.005402  # Highest risk
        elif 8 <= hour < 12:
            return 0.002177
        elif 12 <= hour < 16:
            return 0.001443
        elif 16 <= hour < 20:
            return 0.001488
        elif 20 <= hour < 24:
            return 0.001237  # Lowest risk
        else:
            return 0.002

    logger.info("Helper functions defined")

    def add_time_features(self, data_x):
        logger.info("Add time features")
        data_x = data_x.copy()

        data_x['hour_of_day'] = ((data_x['Time'] % 86400) / 3600).astype(int)
        logger.info("Added hour_of_day")

        data_x['hour_risk_score'] = data_x['hour_of_day'].apply(self.get_hour_risk_score)
        logger.info("Added hour_risk_score")

        data_x['time_normalized'] = data_x['Time'] / data_x['Time'].max()
        logger.info("Added time_normalized")

        return data_x

    def add_amount_features(self, data_x):
        logger.info("Add amount features")
        data_x = data_x.copy()

        scaler = StandardScaler()
        if 'Amount' in data_x.columns:
            amount_scaled = scaler.fit_transform(data_x[['Amount']])
            data_x['amount_z_score'] = amount_scaled.flatten()
        logger.info("Added amount_z_score")

        data_x['is_small_amount'] = (data_x['Amount'] < 10).astype(int)
        data_x['is_very_small_amount'] = (data_x['Amount'] < 5).astype(int)
        logger.info("Added is_small_amount, is_very_small_amount")

        data_x['is_large_amount'] = (data_x['Amount'] > 200).astype(int)
        data_x['is_very_large_amount'] = (data_x['Amount'] > 500).astype(int)
        logger.info("Added is_large_amount, is_very_large_amount")

        data_x['distance_from_fraud_median'] = np.abs(data_x['Amount'] - self.fraud_amount_median)
        data_x['distance_from_fraud_mean'] = np.abs(data_x['Amount'] - self.fraud_amount_mean)
        logger.info("Added distance_from_fraud_median, distance_from_fraud_mean")

        data_x['in_small_fraud_zone'] = ((data_x['Amount'] >= 5) & (data_x['Amount'] <= 15)).astype(int)
        data_x['in_large_fraud_zone'] = ((data_x['Amount'] >= 100) & (data_x['Amount'] <= 300)).astype(int)
        logger.info("Added in_small_fraud_zone, in_large_fraud_zone")

        data_x['fraud_amount_similarity'] = np.minimum(
            1 / (1 + data_x['distance_from_fraud_median']),
            1 / (1 + data_x['distance_from_fraud_mean'])
        )
        logger.info("Added fraud_amount_similarity")

        return data_x, scaler

    def process(self, data_x):
        try:
            logger.info("Features processing")
            data_time_x = self.add_time_features(data_x)
            data_time_amount_x, amount_scaler = self.add_amount_features(data_time_x)
            logger.info("Features processing complete")

            return data_time_amount_x
        except Exception as e:
            logger.error("Error while processing Feature engineering")
            raise e

    def clean_features(self, df):
        try:
            features_to_scale = self.config_loader.config["preprocessor"]["features_to_scale"]
            features_to_keep = self.config_loader.config["preprocessor"]["features_to_keep"]
            all_features = features_to_scale + features_to_keep

            cleaned_df = df[all_features]
            return cleaned_df
        except Exception as e:
            logger.error("Unexpected error: ", e)
            raise e