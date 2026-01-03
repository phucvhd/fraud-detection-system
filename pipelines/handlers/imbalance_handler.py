import logging

from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from pandas import DataFrame

from config.config_loader import ConfigLoader

logger = logging.getLogger(__name__)

class ImbalanceHandler:
    def __init__(self, config_loader: ConfigLoader):
        self.imbalance_config = config_loader.config["imbalance"]
        self.sampling_strategy = self.imbalance_config["sampling_strategy"]
        self.random_state = self.imbalance_config["random_state"]
        self.sampler = None

    def build_sampler(self):
        try:
            method = self.imbalance_config["method"]

            logger.info(f"Building {method} sampler")
            logger.info(f"Sampling strategy: {self.sampling_strategy}")

            if method == "smote":
                self.sampler = SMOTE(
                    sampling_strategy=self.sampling_strategy,
                    random_state=42
                )
            elif method == "random_oversample":
                self.sampler = RandomOverSampler(
                    sampling_strategy=self.sampling_strategy,
                    random_state=42
                )
            elif method == "random_undersample":
                self.sampler = RandomUnderSampler(
                    sampling_strategy=self.sampling_strategy,
                    random_state=42
                )
            else:
                raise ValueError(f"Unknown sampling method: {method}")

            logger.info(f"Sampler built: {method}")

            return self.sampler
        except Exception as e:
            logger.error("Unexpected error: ", e)
            raise e

    def fit_resample(self, data_x: DataFrame, data_y: DataFrame):
        try:
            if self.sampler is None:
                self.build_sampler()

            logger.info("Applying sampling strategy")
            logger.info(f"Before - Total: {len(data_y):,}, Fraud: {data_y.sum():,}, "
                        f"Ratio: {data_x.mean() * 100:.2f}%")

            data_x_resampled, data_y_resampled = self.sampler.fit_resample(data_x, data_y)

            logger.info(f"After - Total: {len(data_y_resampled):,}, "
                        f"Fraud: {data_y_resampled.sum():,}, "
                        f"Ratio: {data_y_resampled.mean() * 100:.2f}%")
            logger.info("Sampling complete")

            return data_x_resampled, data_y_resampled
        except Exception as e:
            logger.error("Unexpected error: ", e)
            raise e