import logging

from imblearn.over_sampling import SMOTE
from pandas import DataFrame

logger = logging.getLogger(__name__)

class Handler:
    def __init__(self, sampling_strategy: float):
        self.sampling_strategy = sampling_strategy

    def smote_fit_resample(self, data_x: DataFrame, data_y: DataFrame) -> (DataFrame, DataFrame):
        try:
            sampler = SMOTE(sampling_strategy=self.sampling_strategy,
                            random_state=42
                            )
            logger.info("Applying sampling strategy")
            logger.info(f"Before - Total: {len(data_y):,}, Fraud: {data_y.sum():,}, "
                        f"Ratio: {data_x.mean() * 100:.2f}%")

            data_x_resampled, data_y_resampled = sampler.fit_resample(data_x, data_y)

            logger.info(f"After - Total: {len(data_y_resampled):,}, "
                        f"Fraud: {data_y_resampled.sum():,}, "
                        f"Ratio: {data_y_resampled.mean() * 100:.2f}%")
            logger.info("Sampling complete")

            return data_x_resampled, data_y_resampled
        except Exception as e:
            logger.error("Unexpected error: ", e)
            raise e