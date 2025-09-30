import logging

import numpy as np
from pandas import DataFrame

logger = logging.getLogger(__name__)

class Validator:
    def __init__(self):
        self._VALID_COLUMNS = ['Time'] + [f"V{i}" for i in range(1, 29)] + ['Amount', 'Class']

    def validate_consistency(self, df: DataFrame) -> bool:
        try:
            if df.empty or df.shape[1] != 31:
                logger.warning("Data is empty or incorrect shape")
                return False

            all_numeric = df.dtypes.apply(lambda x: np.issubdtype(x, np.number)).all()

            if not all_numeric:
                logger.warning("Not all columns are numeric:")
                non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
                logger.warning(f"Non-numeric columns: {non_numeric_cols}")
                return False

            if df.columns.tolist() != self._VALID_COLUMNS:
                logger.warning("Data columns are incorrect")
                return False

            logger.info("Data consistency is valid")
            return True
        except Exception as e:
            logger.error("Unexpected error: ", e)
            raise e

    def validate_integrity(self, df: DataFrame) -> bool:
        try:
            missing_values = df.isnull().sum().sum()

            if missing_values == 0:
                return True

            logger.warning(f"{missing_values} total missing values")

            if missing_values > (df.size * 0.1):
                logger.error("Missing values exceed 10%")

            missing_cols = df.columns[df.isnull().any()]
            logger.warning(f"Columns with missing values: {missing_cols.tolist()}")
            return False
        except Exception as e:
            logger.error("Unexpected error: ", e)
            return False

    def fill_column_missing_values(self, df: DataFrame, column: str) -> bool:
        try:
            value_mean = df[column].mean()
            df[column] = df[column].fillna(value_mean)
            return True
        except Exception as e:
            return False
