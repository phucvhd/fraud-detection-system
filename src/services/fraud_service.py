import io
import json
import logging
import tarfile

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

    def _load_model(self, model_id: str):
        try:
            logger.info(f"Loading model with id={model_id}")
            obj = self.s3_client.get_object(f"models/{model_id}.tar.gz")
            bytestream = io.BytesIO(obj)

            tar = tarfile.open(fileobj=bytestream, mode="r:gz")

            model_joblib = None
            for member in tar.getmembers():
                if member.name.endswith("model.joblib"):
                    model_joblib = member
                    break

            if model_joblib is None:
                raise FileNotFoundError("model.joblib not found inside tar.gz")

            model_file = tar.extractfile(model_joblib)
            model = joblib.load(model_file)

            logger.info(f"Model id={model_id} load successfully")
            return model
        except Exception as e:
            logger.error(f"Failed to load model id={model_id}")
            raise e

    def predict_transaction(self, transaction: dict) -> dict:
        transaction_df = pd.DataFrame([transaction])
        transaction_id = transaction["transaction_id"]
        logger.info(f"Processing transaction={transaction_id}")

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
        logger.info(f"Predict transaction={transaction_id} successfully")
        return result

    def fraud_handler(self, msg_value):
        kafka_service = KafkaService(self.kafka_config_loader)

        transaction = json.loads(msg_value.decode("utf-8"))
        decision = self.predict_transaction(transaction)

        if decision["is_fraud"]:
            fraud_alerts_topic = self.fraud_detection_config["kafka"]["fraud_alerts_topic"]
            kafka_service.send_message(fraud_alerts_topic, decision["transaction_id"], str(decision))
            logger.info(f"An alert for transaction={decision['transaction_id']} has been sent to {fraud_alerts_topic}")

        decision_topic = self.fraud_detection_config["kafka"]["decision_topic"]
        kafka_service.send_message(decision_topic, decision["transaction_id"], str(decision))
        logger.info(f"Decision for transaction={decision['transaction_id']} has been sent to {decision_topic}")

    def _get_confidence_level(self, probability: float) -> str:
        if probability >= 0.8 or probability <= 0.2:
            return "high"
        elif probability >= 0.6 or probability <= 0.4:
            return "medium"
        else:
            return "low"

    def get_hour_risk_score(self, hour):
        if 0 <= hour < 4:
            return 0.003968
        elif 4 <= hour < 8:
            return 0.005402
        elif 8 <= hour < 12:
            return 0.002177
        elif 12 <= hour < 16:
            return 0.001443
        elif 16 <= hour < 20:
            return 0.001488
        elif 20 <= hour < 24:
            return 0.001237
        else:
            return 0.002

    logger.info("Helper functions defined")

    def add_time_features(self, data_x):
        logger.info("Add time-based features")
        data_x = data_x.copy()
        if "Time" in data_x.columns:
            logger.debug("Add time features")
            data_x = data_x.copy()

            data_x["hour_of_day"] = (data_x["Time"] / 3600) % 24
            logger.debug("Added hour_of_day")

            hour = (data_x["Time"] / 3600) % 24
            data_x["day_period"] = pd.cut(hour, bins=[0, 6, 12, 18, 24],
                                     labels=[0, 1, 2, 3], include_lowest=True)
            logger.debug("Added day_period")

            data_x["time_since_start"] = data_x["Time"] / data_x["Time"].max()
            logger.debug("Added time_since_start")
        else:
            logger.error("Missing column=Time")
            raise ValueError("Transaction format is invalid")

        return data_x

    def add_amount_features(self, data_x):
        logger.info("Add amount-based features")
        data_x = data_x.copy()

        scaler = StandardScaler()
        if "Amount" in data_x.columns:
            data_x["log_amount"] = np.log1p(data_x["Amount"])
            logger.debug("Added log_amount")

            data_x["amount_scaled"] = scaler.fit_transform(data_x[["Amount"]])
            logger.debug("Added amount_scaled")
        else:
            logger.error("Missing column=Amount")
            raise ValueError("Transaction format is invalid")

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
            all_features = ["Time"] + features_to_scale + ["Amount", "hour_of_day", "day_period", "time_since_start", "log_amount", "amount_scaled"]

            cleaned_df = df[all_features]
            logger.info(f"Total features: {cleaned_df.shape[1]}")
            return cleaned_df
        except Exception as e:
            logger.error("Unexpected error: ", e)
            raise e