import logging

from pandas import DataFrame
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

class Preprocessor:
    def __init__(self, features_to_scale: list[str], features_to_keep: list[str]):
        self.features_to_scale = features_to_scale
        self.features_to_keep = features_to_keep

    def fit_transform(self, data: DataFrame):
        transformer = ColumnTransformer(
            transformers=[
                ("scaler", StandardScaler(), self.features_to_scale),
                ("passthrough", "passthrough", self.features_to_keep)
            ],
            remainder="drop"
        )

        transformer.fit(data)
        return transformer.transform(data)

    def split_data(self, df: DataFrame, train_ratio: float) -> (DataFrame, DataFrame):
        try:
            logger.info("Splitting data by time")
            df_sorted = df.sort_values("Time").reset_index(drop=True)

            train_size = int(train_ratio * len(df_sorted))

            train_data = df_sorted[:train_size]
            val_data = df_sorted[train_size:]

            logger.info(f"Training: {len(train_data)}")
            logger.info(f"Validation: {len(val_data)}")
            return train_data, val_data
        except Exception as e:
            logger.error("Unexpected error: ", e)
            raise e