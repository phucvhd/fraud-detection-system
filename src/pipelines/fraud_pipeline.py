import json
import logging
from datetime import datetime
from pathlib import Path

import joblib
import mlflow
import pandas as pd

from config.config_loader import ConfigLoader
from src.evaluators.fraud_model_evaluator import FraudModelEvaluator
from src.handlers.imbalance_handler import ImbalanceHandler
from src.loaders.data_loader import DataLoader
from src.loaders.s3_loader import S3Client
from src.preprocessors.fraud_preprocessor import FraudPreprocessor
from src.trainers.fraud_model_trainer import FraudModelTrainer
from src.tuner.hyper_tuner import HyperTuner
from src.validators.fraud_validator import FraudValidator

logger = logging.getLogger(__name__)

class FraudPipeline:
    def __init__(self, config_loader: ConfigLoader):
        self.config_loader = config_loader
        self.raw_data_path = self.config_loader.config["pipeline"]["data"]["raw"]["local"]

        mlflow.set_tracking_uri(self.config_loader.config["mlflow"]["url"])
        mlflow.set_registry_uri(self.config_loader.config["mlflow"]["url"])
        self.model_type = self.config_loader.config["model"]["type"]
        self.data_loader = DataLoader(self.config_loader)
        self.data_validator = FraudValidator()
        self.preprocessor = FraudPreprocessor(self.config_loader)
        self.imbalance_handler = ImbalanceHandler(self.config_loader)
        self.model_trainer = FraudModelTrainer(self.config_loader)
        self.evaluator = FraudModelEvaluator(self.config_loader)
        self.hyper_tuner = HyperTuner(self.config_loader)

        self.run_id = self.generate_time_str()
        self.raw_data = None
        self.raw_data_is_valid = False
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

    def load_data(self):
        self.raw_data = self.data_loader.load_data(self.raw_data_path)
        return self.raw_data

    def load_data_v2(self):
        s3_key = self.config_loader.config["pipeline"]["data"]["raw"]["s3"]
        s3_client = S3Client(self.config_loader)
        local_path = Path(self.raw_data_path)
        try:
            if local_path.exists():
                logger.info(f"Local file found: {local_path}")
                self.raw_data = pd.read_csv(local_path)
                logger.info(f"Loaded from local")
                return self.raw_data
            else:
                logger.info(f"Local file not found: {local_path}")
                logger.info(f"Downloading from S3: {s3_key}")

                local_path.parent.mkdir(parents=True, exist_ok=True)

                s3_client.download_file(s3_key, str(local_path))

                self.raw_data = pd.read_csv(local_path)
                logger.info(f"Downloaded and loaded from S3")
                return self.raw_data

        except Exception as e:
            logger.error(f"✗ Failed to load data: {e}")
            raise

    def validate_data(self):
        if self.raw_data is None:
            raise ValueError("Raw Data must be loaded first")
        if not self.data_validator.validate_quality(self.raw_data):
            raise ValueError("Data validation failed")

        self.raw_data_is_valid = True
        return self.raw_data

    def split_data(self):
        if not self.raw_data_is_valid:
            raise ValueError("Data is in valid")

        self.train_df, self.val_df, self.test_df = self.data_loader.split_train_data(self.raw_data)
        # self.data_loader.save_splits(self.train_df, self.val_df, self.test_df)

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

    def hyper_tuning_v2(self):
        search = self.hyper_tuner.init_tuner()
        search_method = type(search).__name__

        mlflow.log_param("imbalance_sampler", type(self.imbalance_handler.sampler).__name__)
        mlflow.log_param("sampling_strategy", self.imbalance_handler.sampling_strategy)
        mlflow.log_param("model_type", self.model_type)
        mlflow.log_param("search_method", search_method)
        mlflow.log_param("cv", search.cv)

        params = None
        if search_method == "GridSearchCV":
            params = search.param_grid
        elif search_method == "RandomizedSearchCV":
            params = search.param_distributions
        for param, values in params.items():
            mlflow.log_param(param, values)

        search.fit(self.train_x_balanced, self.train_y_balanced)
        for idx in range(len(search.cv_results_['params'])):
            with mlflow.start_run(run_name=f"cv_trial_{idx}", nested=True):
                params = search.cv_results_['params'][idx]
                mlflow.log_params(params)

                for key in search.cv_results_.keys():
                    if 'test' in key:
                        mlflow.log_metric(key, search.cv_results_[key][idx])

        self.best_estimator = search.best_estimator_
        self.best_params = search.best_params_
        self.best_score = search.best_score_

    def train_model(self):
        self.model_trainer.train(self.train_x_balanced, self.train_y_balanced)
        self.model = self.model_trainer.model

    def evaluate_hyper_tune(self):
        if self.best_estimator is None:
            logger.error("Best estimator is not identified")
            return
        score = self.best_estimator.score(self.val_x_processed, self.val_y)
        logger.info(f"Best estimator score: {score}")

    def evaluate_hyper_tune_v2(self):
        if self.best_estimator is None:
            logger.error("Best estimator is not identified")
            return

        self.val_metrics = self.evaluator.evaluate(
            self.best_estimator,
            self.val_x_processed,
            self.val_y
        )

        mlflow.log_params({f"best_{k}": v for k, v in self.best_params.items()})
        mlflow.log_metrics({
            "val_recall": self.val_metrics.recall,
            "val_precision": self.val_metrics.precision,
            "val_f1": self.val_metrics.f1,
            "val_pr_auc": self.val_metrics.pr_auc,
            "cv_best_recall": self.best_score
        })
        self.log_metrics(self.val_metrics)

        if self.model_type == 'xgboost':
            mlflow.xgboost.log_model(self.best_estimator, f"best_model_{self.run_id}")
        else:
            mlflow.sklearn.log_model(self.best_estimator, f"best_model_{self.run_id}")

        logger.info(f"Best CV Recall: {self.best_score}")
        logger.info(f"Validation Recall: {self.val_metrics.recall}")
        logger.info(f"Best Parameters: {self.best_params}")


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

    def evaluate_model_v2(self):
        self.test_metrics = self.evaluator.evaluate(
            self.best_estimator,
            self.test_x_processed,
            self.test_y
        )
        self.log_metrics(self.test_metrics)

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
            mlflow.xgboost.log_model(self.model, name=f"fraud_model_{self.run_id}")
        else:
            mlflow.sklearn.log_model(self.model, name=f"fraud_model_{self.run_id}")

    def run_mlflow_experiment(self):
        try:
            logger.info("MlFlow: Training Fraud detection model")

            experiment_name = "Fraud detection experiments"
            mlflow.set_experiment(experiment_name)

            with mlflow.start_run(run_name=f"experiment_{self.run_id}"):
                self.load_data()
                self.validate_data()
                self.split_data()
                self.preprocess_data()
                self.handle_imbalance()
                self.train_model()
                self.evaluate_model()

                self.log_model()
                self.log_params()
                self.log_metrics(self.val_metrics)

            logger.info("PIPELINE COMPLETE!")

        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise e

    def run_mlflow_experiment_v2(self):
        try:
            logger.info(f"Running experiment_{self.run_id}")

            experiment_name = "Fraud detection experiments"
            mlflow.set_experiment(experiment_name)

            with mlflow.start_run(run_name=f"experiment_{self.run_id}"):
                self.load_data_v2()
                self.validate_data()
                self.split_data()
                self.preprocess_data()
                self.handle_imbalance()
                self.train_model()
                self.evaluate_model()

                self.log_model()
                self.log_params()
                self.log_metrics(self.val_metrics)

            logger.info("PIPELINE COMPLETE!")

        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise e

    def run_mlflow_hyper_tune(self):
        try:
            logger.info("MlFlow: Hyper tuning Fraud detection model")
            mlflow.set_experiment("Fraud detection hyper tuning")

            with mlflow.start_run(run_name=f"Tuning_{self.run_id}"):
                self.load_data()
                self.validate_data()
                self.split_data()
                self.preprocess_data()
                self.handle_imbalance()
                self.hyper_tuning_v2()
                self.evaluate_hyper_tune_v2()

            logger.info("HYPER TUNING COMPLETE!")

        except Exception as e:
            logger.error(f"Hyper tuning failed: {e}")
            raise e

    def generate_time_str(self) -> str:
        now = datetime.now()
        return now.strftime("%Y%m%d_%H%M%S")