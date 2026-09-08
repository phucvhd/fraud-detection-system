import datetime
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
from src.schemas.transaction import TransactionBase, TransactionCanonical
from src.services.kafka_service import KafkaService

logger = logging.getLogger(__name__)


class FraudService:
    def __init__(self, config_loader: ConfigLoader):
        self.config_loader = config_loader
        self.fraud_detection_config = config_loader.config["api"]["fraud_detection"]
        kafka_config_loader = KafkaConfigLoader(config_loader)
        self.kafka_service = KafkaService(kafka_config_loader)
        self.s3_client = S3Client(config_loader)
        self.scaler: StandardScaler | None = None
        self.model = self._load_model(config_loader.config["api"]["fraud_detection"]["model"]["id"])

    def _load_model(self, model_id: str):
        logger.info("Loading model id=%s", model_id)
        try:
            obj = self.s3_client.get_object(f"models/{model_id}.tar.gz")
            tar = tarfile.open(fileobj=io.BytesIO(obj), mode="r:gz")

            model_member = next((m for m in tar.getmembers() if m.name.endswith("model.joblib")), None)
            scaler_member = next((m for m in tar.getmembers() if m.name.endswith("scaler.joblib")), None)

            if model_member is None:
                raise FileNotFoundError("model.joblib not found inside tar.gz")

            model = joblib.load(tar.extractfile(model_member))

            if scaler_member is not None:
                self.scaler = joblib.load(tar.extractfile(scaler_member))
                logger.info("Scaler loaded for model id=%s", model_id)
            else:
                logger.warning("No scaler.joblib found for model id=%s; amount_scaled will use log_amount", model_id)

            logger.info("Model id=%s loaded successfully", model_id)
            return model
        except Exception:
            logger.error("Failed to load model id=%s", model_id, exc_info=True)
            raise

    def predict_transactions(self, transactions: list[dict]) -> list[TransactionCanonical]:
        """Score a batch of transactions with one model call instead of one per
        transaction — DataFrame construction and inference each carry a fixed
        per-call cost that batching amortizes across the whole batch.

        Uses predict_proba() rather than predict(): for a tree-ensemble
        classifier, predict() computes proba internally and argmaxes it anyway,
        so calling predict_proba() once gets both the decision and the
        probability for free instead of paying for two passes.
        """
        if not transactions:
            return []

        logger.info("Processing batch of %d transaction(s)", len(transactions))

        df = pd.DataFrame(transactions)
        df = self.process(df)
        features = self.clean_features(df)

        try:
            # [:, 1] is the probability of the positive (fraud) class.
            probabilities = self.model.predict_proba(features)[:, 1]
            # Matches predict()'s own tie-break (argmax picks the first/lower
            # index), so a 0.5/0.5 split still resolves to "not fraud".
            is_fraud_flags = probabilities > 0.5
        except AttributeError:
            # Model doesn't support predict_proba — fall back to a bare decision
            # with no probability rather than fabricating one.
            is_fraud_flags = self.model.predict(features)
            probabilities = [None] * len(transactions)

        now = datetime.datetime.now(datetime.timezone.utc)
        results = []
        for i, transaction in enumerate(transactions):
            transaction_id = transaction["transaction_id"]
            probability = probabilities[i]
            result = TransactionCanonical(
                transaction_id=transaction_id,
                is_fraud=bool(is_fraud_flags[i]),
                fraud_probability=(float(probability) if probability is not None else None),
                event_time_seconds=transaction["Time"],
                amount=transaction["Amount"],
                event_timestamp=now,
                data_source="ms-fraud-detection",
                created_at=now,
                features={f"V{j}": df.iloc[i][f"V{j}"] for j in range(1, 29)},
            )
            results.append(result)
            logger.info("Predicted transaction=%s is_fraud=%s", transaction_id, result.is_fraud)
        return results

    def predict_transaction(self, transaction: dict) -> TransactionCanonical:
        return self.predict_transactions([transaction])[0]

    def fraud_handler(self, msg_values: list[bytes]) -> None:
        transactions = [json.loads(v.decode("utf-8")) for v in msg_values]
        decisions = self.predict_transactions(transactions)

        for decision in decisions:
            payload = json.dumps(decision.model_dump(mode="json")).encode("utf-8")
            key = str(decision.transaction_id)

            if decision.is_fraud:
                topic = self.fraud_detection_config["kafka"]["fraud_alerts_topic"]
                self.kafka_service.send_message(topic, key, payload)
                logger.info("Fraud alert sent for transaction=%s to %s", key, topic)

            decision_topic = self.fraud_detection_config["kafka"]["decision_topic"]
            self.kafka_service.send_message(decision_topic, key, payload)
            logger.info("Decision sent for transaction=%s to %s", key, decision_topic)

    def add_time_features(self, data_x: pd.DataFrame) -> pd.DataFrame:
        if "Time" not in data_x.columns:
            raise ValueError("Missing required column: Time")

        data_x = data_x.copy()
        hour = (data_x["Time"] / 3600) % 24
        data_x["hour_of_day"] = hour
        data_x["day_period"] = pd.cut(hour, bins=[0, 6, 12, 18, 24], labels=[0, 1, 2, 3], include_lowest=True)
        # Per-row (Time / Time), not Time / batch-max: this used to run one
        # transaction at a time, where max() was always that row's own Time —
        # dividing by the batch max instead would make a transaction's score
        # depend on which other transactions happened to land in the same
        # Kafka poll batch, which must stay identical to single-row scoring.
        data_x["time_since_start"] = data_x["Time"] / data_x["Time"]
        return data_x

    def add_amount_features(self, data_x: pd.DataFrame) -> pd.DataFrame:
        if "Amount" not in data_x.columns:
            raise ValueError("Missing required column: Amount")

        data_x = data_x.copy()
        data_x["log_amount"] = np.log1p(data_x["Amount"])

        if self.scaler is not None:
            data_x["amount_scaled"] = self.scaler.transform(data_x[["Amount"]])
        else:
            data_x["amount_scaled"] = data_x["log_amount"]

        return data_x

    def process(self, data_x: pd.DataFrame) -> pd.DataFrame:
        try:
            data_x = self.add_time_features(data_x)
            data_x = self.add_amount_features(data_x)
            return data_x
        except Exception:
            logger.error("Feature engineering failed", exc_info=True)
            raise

    def clean_features(self, df: pd.DataFrame) -> pd.DataFrame:
        v_features = self.config_loader.config["preprocessor"]["features_to_scale"]
        engineered = ["Amount", "hour_of_day", "day_period", "time_since_start", "log_amount", "amount_scaled"]
        all_features = ["Time"] + v_features + engineered
        return df[all_features]

    def _get_confidence_level(self, probability: float) -> str:
        if probability >= 0.8 or probability <= 0.2:
            return "high"
        elif probability >= 0.6 or probability <= 0.4:
            return "medium"
        return "low"

    def get_hour_risk_score(self, hour: int) -> float:
        risk_map = {
            range(0, 4): 0.003968,
            range(4, 8): 0.005402,
            range(8, 12): 0.002177,
            range(12, 16): 0.001443,
            range(16, 20): 0.001488,
            range(20, 24): 0.001237,
        }
        return next((v for r, v in risk_map.items() if hour in r), 0.002)
