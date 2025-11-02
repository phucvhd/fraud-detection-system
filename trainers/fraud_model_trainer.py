import json
import logging
from datetime import datetime
from pathlib import Path

import joblib
import mlflow
import pandas as pd
import numpy as np
from pandas import DataFrame, Series

from config.config_loader import ConfigLoader
from services.evaluators.fraud_model_evaluator import FraudModelEvaluator
from services.handlers.imbalance_handler import ImbalanceHandler
from services.loaders.data_loader import DataLoader
from services.model_trainers.fraud_model_trainer import FraudModel
from services.preprocessors.fraud_preprocessor import FraudPreprocessor
from services.tuner.hyper_tuner import HyperTuner
from services.validators.fraud_validator import FraudValidator

logger = logging.getLogger(__name__)

class FraudModelTrainer:
    def __init__(self, config_loader: ConfigLoader):
        self.config_loader = config_loader
        self.raw_data_path = "../data/raw/creditcard.csv"

        self.model_type = self.config_loader.config["model"]["type"]
        self.data_loader = DataLoader(self.config_loader)
        self.data_validator = FraudValidator()
        self.preprocessor = FraudPreprocessor(self.config_loader)
        self.imbalance_handler = ImbalanceHandler(self.config_loader)
        self.model_trainer = FraudModel(self.config_loader)
        self.evaluator = FraudModelEvaluator(self.config_loader)
        self.hyper_tuner = HyperTuner(self.config_loader)

        self.run_id = self.generate_time_str()
        self.train_df = None
        self.val_df = None
        self.test_df = None
        self.model = None
        self.test_y = None
        self.val_y = None
        self.train_y = None
        self.test_x_processed = None
        self.val_x_processed = None
        self.train_x_processed = None
        self.train_y_balanced = None
        self.train_x_balanced = None
        self.test_metrics = None
        self.optimal_threshold = None
        self.val_metrics = None
        self.best_params = None
        self.best_score = None
        self.cv_results = None
        self.best_estimator = None

    def load_and_split_data(self):
        df = self.data_loader.load_data(self.raw_data_path)

        if not self.data_validator.validate_quality(df):
            raise ValueError("Data validation failed")

        self.train_df, self.val_df, self.test_df = self.data_loader.split_train_data(df)

        self.data_loader.save_splits(self.train_df, self.val_df, self.test_df)

    def preprocess_data(self):
        train_x, train_y = self.data_loader.seperate_fetures(self.train_df)
        val_x, val_y = self.data_loader.seperate_fetures(self.val_df)
        test_x, test_y = self.data_loader.seperate_fetures(self.test_df)

        self.train_x_processed = self.preprocessor.fit_transform(train_x)
        self.val_x_processed = self.preprocessor.transform(val_x)
        self.test_x_processed = self.preprocessor.transform(test_x)

        self.train_y = train_y
        self.val_y = val_y
        self.test_y = test_y

        logger.info(f"Preprocessing complete")

    def handle_imbalance(self):
        (self.train_x_balanced,
         self.train_y_balanced) = self.imbalance_handler.fit_resample(self.train_x_processed, self.train_y)

    def hyper_tuning(self):
        search = self.hyper_tuner.init_tuner()
        search.fit(self.train_x_balanced, self.train_y_balanced)
        self.best_params = search.best_params_
        self.best_score = search.best_score_
        self.cv_results = search.cv_results_
        self.best_estimator = search.best_estimator_
        logger.info(f"Best {self.model_type} params: {self.best_params}")
        logger.info(f"Best CV recall: {self.best_score:.4f}")

    def train_model(self):
        self.model_trainer.train(self.train_x_balanced, self.train_y_balanced)
        self.model = self.model_trainer.model

    def evaluate_hyper_tune(self):
        if self.best_estimator is None:
            logger.error("Best estimator is not identified")
            return
        score = self.best_estimator.score(self.val_x_processed, self.val_y)
        logger.info(f"Best estimator score: {score}")

    def evaluate_model(self):
        self.val_metrics = self.evaluator.evaluate(
            self.model,
            self.val_x_processed,
            self.val_y
        )

        self.optimal_threshold = self.evaluator.find_optimal_threshold(
            self.model,
            self.val_x_processed,
            self.val_y
        )

        self.test_metrics = self.evaluator.evaluate(
            self.model,
            self.test_x_processed,
            self.test_y
        )

    def save_artifacts(self):
        now = datetime.now()
        time_str = now.strftime("%Y%m%d_%H%M%S")
        output_path = Path(self.config_loader.config["output"]["model_path"] + f"model_{time_str}")
        output_path.mkdir(parents=True, exist_ok=True)

        joblib.dump(
            self.preprocessor.pipeline,
            output_path / "preprocessing_pipeline.pkl"
        )
        logger.info("Preprocessing pipeline saved")

        joblib.dump(
            self.model,
            output_path / "fraud_model.pkl"
        )
        logger.info("Model saved")

        feature_names = self.preprocessor.get_feature_names()
        joblib.dump(
            feature_names,
            output_path / "feature_names.pkl"
        )
        logger.info("Feature names saved")

        metadata = {
            "training_date": datetime.now().isoformat(),
            "model_type": self.model_type,
            "model_params": self.config_loader.config["model"]["params"][self.model_type],
            "optimal_threshold": float(self.optimal_threshold),
            "validation_metrics": {
                k: float(v) if isinstance(v, (int, float)) else str(v)
                for k, v in self.val_metrics.items()
                if k != "confusion_matrix"
            },
            "test_metrics": {
                k: float(v) if isinstance(v, (int, float)) else str(v)
                for k, v in self.test_metrics.items()
                if k != "confusion_matrix"
            }
        }

        with open(output_path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info("Metadata saved")
        logger.info(f"All artifacts saved to: {output_path}")

    def save_report(self):
        output_path = Path(self.config_loader.config["output"]["report_path"])
        output_path.mkdir(parents=True, exist_ok=True)
        training_date = datetime.now().isoformat()

        df = pd.DataFrame([
            {
                "dataset": "validation",
                "training_date": training_date,
                "imbalance_method": self.config_loader.config["imbalance"]["method"],
                "sampling_strategy": self.config_loader.config["imbalance"]["sampling_strategy"],
                "model_type": self.model_type,
                **self.config_loader.config["model"]["params"][self.model_type],
                "optimal_threshold": float(self.optimal_threshold),
                **self.val_metrics
            },
            {
                "dataset": "test",
                "training_date": training_date,
                "imbalance_method": self.config_loader.config["imbalance"]["method"],
                "sampling_strategy": self.config_loader.config["imbalance"]["sampling_strategy"],
                "model_type": self.model_type,
                **self.config_loader.config["model"]["params"][self.model_type],
                "optimal_threshold": float(self.optimal_threshold),
                **self.test_metrics
            }
        ])

        file_name = f"fraud_model_report_{self.run_id}.csv"
        df.to_csv(output_path / file_name, index=False)
        logger.info(f"Report saved to: {output_path}")

    def log_params(self) -> None:
        logger.info("Logging parameters to Mlflow")
        mlflow.log_params(self.config_loader.
                          config["model"]["params"][self.model_type])

    def log_metrics(self, metrics: dict[str, float]) -> None:
        logger.info("Logging metrics to Mlflow")
        mlflow.log_metrics(metrics)

    def log_model(self) -> None:
        logger.info("Logging model to Mlflow")
        if self.model is None:
            raise ValueError("Model must be trained before logging")

        if self.model_type == 'xgboost':
            mlflow.xgboost.log_model(self.model, name="model")
        else:
            mlflow.sklearn.log_model(self.model, name="model")

    def train(self, x_train: DataFrame, y_train: Series) -> None:
        self.model.fit(x_train, y_train)

    def predict(self, x: DataFrame) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model must be trained before prediction")
        return self.model.predict(x)

    def predict_proba(self, x: DataFrame) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model must be trained before prediction")
        return self.model.predict_proba(x)[:, 1]

    def run(self):
        try:
            experiment_name = f"fraud_detection_experiments_{self.run_id}"
            mlflow.set_experiment(experiment_name)

            with mlflow.start_run(run_name="my_experiment"):
                self.load_and_split_data()
                self.preprocess_data()
                self.handle_imbalance()
                self.train_model()
                self.evaluate_model()
                self.save_artifacts()
                self.save_report()

                self.log_model()
                self.log_params()
                metrics = {k: v for k, v in self.val_metrics.items() if k != 'confusion_matrix'}
                self.log_metrics(metrics)

            logger.info("PIPELINE COMPLETE!")

        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise e

    def generate_time_str(self) -> str:
        now = datetime.now()
        return now.strftime("%Y%m%d_%H%M%S")