import logging

import xgboost as xgb
from pandas import DataFrame
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from config.config_loader import ConfigLoader

logger = logging.getLogger(__name__)

class FraudModelTrainer:
    def __init__(self, config_loader: ConfigLoader):
        self.model_config = config_loader.config["model"]
        self.model = None

    def build_model(self):
        try:
            model_type = self.model_config["type"]
            params = self.model_config["params"]

            logger.info(f"Building {model_type} model")

            if model_type == "xgboost":
                self.model = xgb.XGBClassifier(**params["xgboost"])
            elif model_type == "random_forest":
                self.model = RandomForestClassifier(**params["random_forest"])
            elif model_type == "decision_tree":
                self.model = DecisionTreeClassifier(**params["decision_tree"])
            else:
                raise ValueError(f"Unknown model type: {model_type}")

            logger.info(f"Model built: {model_type}")

            return self.model
        except Exception as e:
            logger.error("Unexpected error: ", e)
            raise e

    def train(self, train_x: DataFrame, train_y: DataFrame):
        if self.model is None:
            self.build_model()

        logger.info("Training model")
        logger.info(f"Training samples: {len(train_x):,}")
        logger.info(f"Fraud samples: {train_y.sum():,}")

        self.model.fit(train_x, train_y)

        logger.info("Model training complete")