import logging

import numpy as np
from sklearn.preprocessing import StandardScaler

from config.config_loader import ConfigLoader

logger = logging.getLogger(__name__)

class FraudFeatureEngineering:
    def __init__(self, config_loader: ConfigLoader):
        self.fraud_amount_median = 9.25
        self.fraud_amount_mean = 122.21

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

    def add_time_features(self, train_x):
        logger.info("Add time features")
        train_x = train_x.copy()

        # Extract hour of day (Time is in seconds)
        train_x['hour_of_day'] = ((train_x['Time'] % 86400) / 3600).astype(int)
        logger.info("Added hour_of_day")

        # Hour-based risk score
        train_x['hour_risk_score'] = train_x['hour_of_day'].apply(self.get_hour_risk_score)
        logger.info("Added hour_risk_score")

        # Time normalized (0 to 1)
        train_x['time_normalized'] = train_x['Time'] / train_x['Time'].max()
        logger.info("Added time_normalized")

        return train_x

    def add_amount_features(self, train_x):
        logger.info("Add amount features")
        train_x = train_x.copy()

        # Fit scaler on training Amount
        scaler = StandardScaler()
        if 'Amount' in train_x.columns:
            amount_scaled = scaler.fit_transform(train_x[['Amount']])
            train_x['amount_z_score'] = amount_scaled.flatten()
        logger.info("Added amount_z_score")

        # Small amount flags
        train_x['is_small_amount'] = (train_x['Amount'] < 10).astype(int)
        train_x['is_very_small_amount'] = (train_x['Amount'] < 5).astype(int)
        logger.info("Added is_small_amount, is_very_small_amount")

        # Large amount flags
        train_x['is_large_amount'] = (train_x['Amount'] > 200).astype(int)
        train_x['is_very_large_amount'] = (train_x['Amount'] > 500).astype(int)
        logger.info("Added is_large_amount, is_very_large_amount")

        # Distance from fraud patterns
        train_x['distance_from_fraud_median'] = np.abs(train_x['Amount'] - self.fraud_amount_median)
        train_x['distance_from_fraud_mean'] = np.abs(train_x['Amount'] - self.fraud_amount_mean)
        logger.info("Added distance_from_fraud_median, distance_from_fraud_mean")

        # Fraud zone flags
        train_x['in_small_fraud_zone'] = ((train_x['Amount'] >= 5) & (train_x['Amount'] <= 15)).astype(int)
        train_x['in_large_fraud_zone'] = ((train_x['Amount'] >= 100) & (train_x['Amount'] <= 300)).astype(int)
        logger.info("Added in_small_fraud_zone, in_large_fraud_zone")

        # Fraud similarity score
        train_x['fraud_amount_similarity'] = np.minimum(
            1 / (1 + train_x['distance_from_fraud_median']),
            1 / (1 + train_x['distance_from_fraud_mean'])
        )
        logger.info("Added fraud_amount_similarity")

        return train_x, scaler

    def process(self, train_x):
        try:
            logger.info("Features processing")
            train_time_x = self.add_time_features(train_x)
            train_time_amount_x, amount_scaler = self.add_amount_features(train_time_x)
            logger.info("Features processing complete")

            return train_time_amount_x
        except Exception as e:
            logger.error("Error while processing Feature engineering")
            raise e

