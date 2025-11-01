import logging
import os
import time
from datetime import datetime

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, confusion_matrix, mean_absolute_error
from sklearn.preprocessing import StandardScaler

from services.handlers.imbalance_handler import ImbalanceHandler
from services.validators.credit_fraud_validator import CreditFraudValidator

logger = logging.getLogger(__name__)

class CreditFraudDetection:
    def __init__(self, train_ratio=0.7, val_ratio=0.15, fraud_ratio=0.2, n_estimators=100):
        self.df = None
        self.fraud_ratio = fraud_ratio
        self.validator = CreditFraudValidator()
        self.imbalance_handler = ImbalanceHandler(sampling_strategy=fraud_ratio)
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.n_estimators = n_estimators
        self._preprocessor = None
        self._train_y = None
        self._train_x = None
        self._train_x_processed = None
        self._val_x_processed = None
        self._test_x_processed = None
        self._trained_model = None
        self.recall = None
        self.precision = None
        self.mae = None
        self.feature_names_out = None
        self.features_to_scale = None
        self.features_to_keep = None

    def load_train_data(self, file_path: str):
        logger.info(f"Loading credit fraud train data from {file_path}")
        self.df = pd.read_csv(file_path)
        logger.info(f"Successfully load credit fraud train data: {len(self.df)} rows")

    def _split_train_data(self):
        df_sorted = self.df.sort_values("Time").reset_index(drop=True)

        train_size = int(self.train_ratio * len(df_sorted))
        val_size = int(self.val_ratio * len(df_sorted))

        self.train_data = df_sorted[:train_size]
        self.val_data = df_sorted[train_size:train_size + val_size]
        self.test_data = df_sorted[train_size + val_size:]

        logger.info(f"Training: {len(self.train_data)}")
        logger.info(f"Validation: {len(self.val_data)}")
        logger.info(f"Test: {len(self.test_data)}")

    def _seperate_fetures(self):
        self._train_x = self.train_data.drop("Class", axis=1)
        self._train_y = self.train_data["Class"]
        self.val_x = self.val_data.drop("Class", axis=1)
        self.val_y = self.val_data["Class"]
        self.test_x = self.test_data.drop("Class", axis=1)
        self.test_y = self.train_data["Class"]

    def _scale_data(self):
        self.features_to_scale = ["Time", "Amount"]
        self.features_to_keep = [col for col in self.train_data.columns if col.startswith("V")]

        self._preprocessor = ColumnTransformer(
            transformers=[
                ("scaler", StandardScaler(), self.features_to_scale),
                ("passthrough", "passthrough", self.features_to_keep)
            ],
            remainder="drop"
        )

        logger.info("Fitting preprocessor on training data...")
        self._preprocessor.fit(self.train_data)

        self._train_x_processed = self._preprocessor.transform(self.train_data)
        self._val_x_processed = self._preprocessor.transform(self.val_data)
        self._test_x_processed = self._preprocessor.transform(self.test_data)

        self.feature_names_out = (
                [f"{feat}_scaled" for feat in self.features_to_scale] +
                self.features_to_keep
        )

        logger.info(f"Preprocessed features: {self.feature_names_out}")
        logger.info(f"Training data shape after preprocessing: {self._train_x_processed.shape}")

    def train_model(self):
        try:
            start_time = time.time()
            if self.df is None:
                raise Exception("Data is not loaded")

            if not self.validator.validate_quality(self.df):
                raise Exception("Data quality is invalid")

            self._split_train_data()

            self._seperate_fetures()

            self._scale_data()

            if not self.validator.validate_imbalance(self.train_data):
                self._train_x, self._train_y = self.imbalance_handler.smote_fit_data(self._train_x_processed, self._train_y)

            rf_model = RandomForestClassifier(n_estimators=self.n_estimators, max_depth=None, random_state=9)
            rf_model.fit(self._train_x, self._train_y)

            self._trained_model = rf_model
            end_time = time.time()
            logger.info(f"Model training complete. Execution time: {(end_time - start_time):.4f} seconds")
        except Exception as e:
            logger.error("Unexpected error: ", e)
            exit(1)

    def evaluate_trained_model(self):
        try:
            if (self._trained_model or self._val_x_processed) is None:
                raise Exception("Model is not trained")
            predicted_val_data_y = self._trained_model.predict(self._val_x_processed)
            self.precision = precision_score(self.val_y, predicted_val_data_y)
            self.recall = recall_score(self.val_y, predicted_val_data_y)
            self.mae = mean_absolute_error(self.val_y, predicted_val_data_y)

            logger.info(f"Precision: {self.precision:.4f}")
            logger.info(f"Recall: {self.recall:.4f}")
            logger.info(f"MAE: {self.mae:.4f}")

            cm = confusion_matrix(self.val_y, predicted_val_data_y)
            logger.info(f"\nConfusion Matrix:")
            logger.info(cm)

        except Exception as e:
            logger.error("Unexpected error: ", e)
            exit(1)

    def save_model(self):
        try:
            if (self._preprocessor
                or self._trained_model
                or self.feature_names_out
                or self.features_to_scale
                or self.features_to_keep
                or self.precision
                or self.recall) is None:
                raise Exception("Model is not ready to save")

            os.makedirs("../models", exist_ok=True)

            joblib.dump(self._preprocessor, "models/preprocessing_pipeline.pkl")
            logger.info("Preprocessing pipeline saved")

            joblib.dump(self._trained_model, "models/fraud_model.pkl")
            logger.info("Model saved")

            joblib.dump(self.feature_names_out, "models/feature_names.pkl")
            logger.info("Feature names saved")

            metadata = {
                "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "model_type": "Random Forest",
                "features_scaled": self.features_to_scale,
                "features_kept": self.features_to_keep,
                "test_precision": self.precision,
                "test_recall": self.recall,
                "fraud_threshold": self.fraud_ratio
            }

            joblib.dump(metadata, "models/model_metadata.pkl")
            logger.info("Metadata saved")

        except Exception as e:
            logger.error("Unexpected error: ", e)
            exit(1)
