import logging

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from config.config_loader import ConfigLoader

logger = logging.getLogger(__name__)

class FraudFeatureEngineering:
    def __init__(self, config_loader: ConfigLoader):
        self.fraud_amount_median = 9.25
        self.fraud_amount_mean = 122.21

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

