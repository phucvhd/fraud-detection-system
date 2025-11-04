import logging

import mlflow

logger = logging.getLogger(__name__)

class Logger:
    def __init__(self, run_id: str):
        self.run_id = run_id

    def log_xgboost_model(self, model) -> None:
        logger.info("Logging xgboost model")
        if model is None:
            raise ValueError("Model must be trained before logging")

        mlflow.xgboost.log_model(model, artifact_path=f"fraud_model_{self.run_id}")

    def log_sklearn_model(self, model) -> None:
        logger.info("Logging sklearn model")
        if model is None:
            raise ValueError("Model must be trained before logging")

        mlflow.sklearn.log_model(model, artifact_path=f"fraud_model_{self.run_id}")

    def log_params(self, params: dict[str, float]) -> None:
        logger.info("Logging parameters")
        mlflow.log_params(params)

    def log_metrics(self, metrics: dict[str, float]) -> None:
        logger.info("Logging metrics")
        mlflow.log_metrics(metrics)
