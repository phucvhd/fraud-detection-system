import logging

import numpy as np
from pandas import DataFrame

logger = logging.getLogger(__name__)

class FraudValidator:
    def __init__(self):
        self._VALID_COLUMNS = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount", "Class"]

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

            negative_times = (df["Time"] < 0).sum()
            if negative_times > 0:
                logger.warning(f"{negative_times} total negative times")
                return False

            negative_amounts = (df["Amount"] < 0).sum()
            if negative_amounts > 0:
                logger.warning(f"{negative_amounts} total negative times")
                return False

            valid_class_range = df["Class"].isin([0, 1]).all()
            if not valid_class_range:
                logger.warning(f"Class range is not valid")
                return False

            logger.info("Data consistency is valid")
            return True
        except Exception as e:
            logger.error("Unexpected error: ", e)
            raise e

    def validate_integrity(self, df: DataFrame) -> bool:
        try:
            missing_values = df.isnull().sum().sum()
            infinite_values = np.isinf(df.select_dtypes(include=[np.number])).sum().sum()

            if missing_values == 0 and infinite_values == 0:
                return True

            if infinite_values > 0:
                logger.warning(f"{infinite_values} total infinite values")
                return False

            logger.warning(f"{missing_values} total missing values")

            if missing_values > (df.size * 0.1):
                logger.error("Missing values exceed 10%")

            missing_cols = df.columns[df.isnull().any()]
            logger.warning(f"Columns with missing values: {missing_cols.tolist()}")
            return False
        except Exception as e:
            logger.error("Unexpected error: ", e)
            return False

    def validate_duplicate(self, df: DataFrame) -> bool:
        try:
            duplicate_rows = df.duplicated().sum()
            if duplicate_rows > 0:
                logger.warning(f"{duplicate_rows} total duplicate rows")
                if duplicate_rows / df.shape[0] > 0.01:
                    logger.warning("Duplicate rows exceed 1%. Risk of misleading analysis")
                    return False
            return True
        except Exception as e:
            logger.error("Unexpected error: ", e)
            return False

    def fill_column_missing_values(self, df: DataFrame, column: str) -> bool:
        try:
            value_mean = df[column].mean()
            df[column] = df[column].fillna(value_mean)
            return True
        except Exception as e:
            logger.error("Unexpected error: ", e)
            return False

    def validate_imbalance(self, df: DataFrame) -> bool:
        try:
            fraud_counts = df["Class"].value_counts()
            fraud_percentage = df["Class"].value_counts(normalize=True) * 100

            logger.info(f"Normal Transactions: {fraud_counts[0]:,} ({fraud_percentage[0]:.3f}%)")
            logger.info(f"Fraudulent Transactions: {fraud_counts[1]:,} ({fraud_percentage[1]:.3f}%)")
            logger.info(f"Fraud Rate: 1 in {int(1 / fraud_percentage[1] * 100)} transactions")

            if 0.3 < fraud_percentage[1] < 0.5:
                logger.info("Data is balanced")
                return True
            if fraud_percentage[1] < 0.1:
                logger.warning("Data is highly imbalanced")
            elif fraud_percentage[1] < 0.01:
                logger.warning("Data is extremely imbalanced")
            else:
                logger.warning("Data is imbalanced")

            return False
        except Exception as e:
            logger.error("Unexpected error: ", e)
            return False

    def validate_quality(self, df: DataFrame) -> bool:
        try:
            if not self.validate_consistency(df):
                return False
            elif not self.validate_integrity(df):
                return False
            elif not self.validate_duplicate(df):
                return False
            else:
                logger.info("Data quality assessment completed. Dataset is ready")
                return True
        except Exception as e:
            logger.error("Unexpected error: ", e)
            return False
