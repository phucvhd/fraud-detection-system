import logging
import random
import uuid

import numpy as np
import pandas as pd
from pandas import DataFrame

from config.config_loader import ConfigLoader

logger = logging.getLogger(__name__)


class FraudSyntheticDataGenerator:
    def __init__(self, config_loader: ConfigLoader, df: DataFrame, seed: int = 42):
        self.max_transaction_time = config_loader.config["fraud_generator"]["max_transaction_time"]
        self.fraud_rate = config_loader.config["fraud_generator"]["fraud_rate"]
        np.random.seed(seed)
        random.seed(seed)
        self.params = self._calibrate_from_dataframe(df)

    def _calibrate_from_dataframe(self, df: DataFrame) -> dict:
        try:
            normal_df = df[df["Class"] == 0]
            fraud_df = df[df["Class"] == 1]

            normal_amounts = normal_df["Amount"]
            normal_amounts_nonzero = normal_amounts[normal_amounts > 0]
            log_amounts = np.log(normal_amounts_nonzero)

            calibrated = {
                "normal": {
                    "amount": {
                        "mean": log_amounts.mean(),
                        "sigma": log_amounts.std(),
                        "min": normal_amounts_nonzero.min(),
                        "max": normal_amounts_nonzero.max(),
                        "p99": normal_amounts_nonzero.quantile(0.99),
                    },
                    "v_features": {},
                },
                "fraud": {
                    "amount_ranges": {},
                    "v_features": {},
                },
            }

            fraud_amounts = fraud_df["Amount"]
            p4, p7 = fraud_amounts.quantile(0.4), fraud_amounts.quantile(0.7)
            buckets = {
                "small": fraud_amounts[fraud_amounts < p4],
                "medium": fraud_amounts[(fraud_amounts >= p4) & (fraud_amounts < p7)],
                "large": fraud_amounts[fraud_amounts >= p7],
            }
            for name, subset in buckets.items():
                calibrated["fraud"]["amount_ranges"][name] = {
                    "proportion": len(subset) / len(fraud_amounts),
                    "min": subset.min(),
                    "max": subset.max(),
                    "mean": subset.mean(),
                }

            for i in range(1, 29):
                col = f"V{i}"
                calibrated["normal"]["v_features"][col] = {
                    "mean": normal_df[col].mean(),
                    "std": normal_df[col].std(),
                }
                calibrated["fraud"]["v_features"][col] = {
                    "mean": fraud_df[col].mean(),
                    "std": fraud_df[col].std(),
                }

            return calibrated
        except Exception:
            logger.error("Failed to calibrate generator from dataframe", exc_info=True)
            raise

    def generate_normal_transaction(self, time_interval: int) -> dict:
        amount_params = self.params["normal"]["amount"]
        data = {
            "transaction_id": str(uuid.uuid4()),
            "Time": time_interval,
            "Amount": np.random.lognormal(mean=amount_params["mean"], sigma=amount_params["sigma"]),
            "Class": 0,
        }
        for i in range(1, 29):
            col = f"V{i}"
            p = self.params["normal"]["v_features"][col]
            data[col] = np.random.normal(p["mean"], p["std"])
        return data

    def generate_normal_transactions(self, n_samples: int = 1000) -> DataFrame:
        time_intervals = np.linspace(0, self.max_transaction_time, num=n_samples).tolist()
        return pd.DataFrame([self.generate_normal_transaction(t) for t in time_intervals])

    def generate_fraudulent_transaction(self, time_interval: int) -> dict:
        ranges = self.params["fraud"]["amount_ranges"]
        bucket_names = list(ranges.keys())
        proportions = [ranges[b]["proportion"] for b in bucket_names]
        chosen = np.random.choice(bucket_names, p=proportions)
        amount = np.random.uniform(ranges[chosen]["min"], ranges[chosen]["max"])

        data = {
            "transaction_id": str(uuid.uuid4()),
            "Time": time_interval,
            "Amount": amount,
            "Class": 1,
        }
        for i in range(1, 29):
            col = f"V{i}"
            p = self.params["fraud"]["v_features"][col]
            data[col] = np.random.normal(p["mean"], p["std"])
        return data

    def generate_fraudulent_transactions(self, n_samples: int = 100) -> DataFrame:
        time_intervals = np.linspace(0, self.max_transaction_time, num=n_samples).tolist()
        return pd.DataFrame([self.generate_fraudulent_transaction(t) for t in time_intervals])

    def generate_transaction(self, time_interval: int) -> dict:
        if random.random() < self.fraud_rate:
            return self.generate_fraudulent_transaction(time_interval)
        return self.generate_normal_transaction(time_interval)

    def generate_dataset(self, n_normal: int = 10000, n_fraud: int = 100, shuffle: bool = True) -> DataFrame:
        try:
            normal_df = self.generate_normal_transactions(n_normal)
            fraud_df = self.generate_fraudulent_transactions(n_fraud)
            combined = pd.concat([normal_df, fraud_df], ignore_index=True)
            if shuffle:
                combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)
            column_order = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount", "Class"]
            return combined[column_order]
        except Exception:
            logger.error("Failed to generate dataset", exc_info=True)
            raise

    def save_dataset(self, df: DataFrame, filename: str = "synthetic_fraud_data.csv") -> None:
        try:
            df.to_csv(filename, index=False)
            logger.info("Dataset saved to %s", filename)
        except Exception:
            logger.error("Failed to save dataset to %s", filename, exc_info=True)
            raise
