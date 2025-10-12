import logging
from pathlib import Path

import pandas as pd
from pandas import DataFrame

from config.config_loader import ConfigLoader

logger = logging.getLogger(__name__)

class DataLoader:
    def __init__(self, config_loader: ConfigLoader):
        self.train_ratio = config_loader.config["model"]["train_ratio"]
        self.val_ratio = config_loader.config["model"]["val_ratio"]

    def load_data(self, file_path: str) -> DataFrame:
        try:
            logger.info(f"Loading raw data from {file_path}")
            df = pd.read_csv(file_path)
            logger.info(f"Successfully load raw data: {len(df)} rows")
            return df
        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
            raise
        except Exception as e:
            logger.error("Unexpected error: ", e)
            raise e

    def split_train_data(self, df: DataFrame):
        try:
            logger.info("Splitting data by time")
            df_sorted = df.sort_values("Time").reset_index(drop=True)

            train_size = int(self.train_ratio * len(df_sorted))
            val_size = int(self.val_ratio * len(df_sorted))

            train_data = df_sorted[:train_size]
            val_data = df_sorted[train_size:train_size + val_size]
            test_data = df_sorted[train_size + val_size:]

            logger.info(f"Training: {len(train_data)}")
            logger.info(f"Validation: {len(val_data)}")
            logger.info(f"Test: {len(test_data)}")
            return train_data, val_data, test_data
        except Exception as e:
            logger.error("Unexpected error: ", e)
            raise e

    def seperate_fetures(self, df: DataFrame):
        try:
            data_x = df.drop("Class", axis=1)
            data_y = df["Class"]
            return data_x, data_y
        except Exception as e:
            logger.error("Unexpected error: ", e)
            raise e

    def save_splits(self, train_df, val_df, test_df, processed_path="../data/processed/"):
        output_path = Path(processed_path)
        output_path.mkdir(parents=True, exist_ok=True)

        logger.info("Saving data splits")

        train_df.to_csv(output_path / 'train.csv', index=False)
        val_df.to_csv(output_path / 'val.csv', index=False)
        test_df.to_csv(output_path / 'test.csv', index=False)

        logger.info(f"Saved splits to {output_path}")