import logging
import random

import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

class FraudSyntheticDataGenerator:
    def __init__(self, df: DataFrame, seed=42):
        self._MAX_TRANSACTION_TIME = 172800
        np.random.seed(seed)
        random.seed(seed)
        self.scaler = StandardScaler()
        self.params = self._calibrate_from_dataframe(df)

    def _calibrate_from_dataframe(self, df: DataFrame):
        try:
            normal_df = df[df['Class'] == 0]
            fraud_df = df[df['Class'] == 1]

            calibrated = {
                'normal': {'v_features': {}},
                'fraud': {'v_features': {}}
            }

            normal_amounts = normal_df['Amount']
            normal_amounts_nonzero = normal_amounts[normal_amounts > 0]
            log_amounts = np.log(normal_amounts_nonzero)

            calibrated['normal']['amount'] = {
                'mean': log_amounts.mean(),
                'sigma': log_amounts.std(),
                'min': normal_amounts_nonzero.min(),
                'max': normal_amounts_nonzero.max(),
                'p99': normal_amounts_nonzero.quantile(0.99)
            }

            fraud_amounts = fraud_df['Amount']

            p4 = fraud_amounts.quantile(0.4)
            p7 = fraud_amounts.quantile(0.7)

            small_frauds = fraud_amounts[fraud_amounts < p4]
            medium_frauds = fraud_amounts[(fraud_amounts >= p4) & (fraud_amounts < p7)]
            large_frauds = fraud_amounts[fraud_amounts >= p7]

            calibrated['fraud']['amount_ranges'] = {
                'small': {
                    'proportion': len(small_frauds) / len(fraud_amounts),
                    'min': small_frauds.min(),
                    'max': small_frauds.max(),
                    'mean': small_frauds.mean()
                },
                'medium': {
                    'proportion': len(medium_frauds) / len(fraud_amounts),
                    'min': medium_frauds.min(),
                    'max': medium_frauds.max(),
                    'mean': medium_frauds.mean()
                },
                'large': {
                    'proportion': len(large_frauds) / len(fraud_amounts),
                    'min': large_frauds.min(),
                    'max': large_frauds.max(),
                    'mean': large_frauds.mean()
                }
            }

            for i in range(1, 29):
                v_col = f'V{i}'

                calibrated['normal']['v_features'][v_col] = {
                    'mean': normal_df[v_col].mean(),
                    'std': normal_df[v_col].std()
                }

                calibrated['fraud']['v_features'][v_col] = {
                    'mean': fraud_df[v_col].mean(),
                    'std': fraud_df[v_col].std()
                }

            self.params = calibrated

            return calibrated
        except Exception as e:
            logger.error("Unexpected error: ", e)
            raise e

    def generate_normal_transactions(self, n_samples=1000) -> DataFrame:
        try:
            data = {'Time': np.sort(np.random.uniform(0, self._MAX_TRANSACTION_TIME, n_samples))}

            amount_params = self.params['normal']['amount']
            data['Amount'] = np.random.lognormal(
                mean=amount_params['mean'],
                sigma=amount_params['sigma'],
                size=n_samples
            )
            data['Amount'] = np.clip(
                data['Amount'],
                amount_params['min'],
                amount_params['p99']
            )

            for i in range(1, 29):
                v_col = f'V{i}'
                v_params = self.params['normal']['v_features'][v_col]

                data[v_col] = np.random.normal(
                    v_params['mean'],
                    v_params['std'],
                    n_samples
                )

            data['Class'] = 0

            return pd.DataFrame(data)
        except Exception as e:
            logger.error("Unexpected error: ", e)
            raise e

    def generate_fraudulent_transactions(self, n_samples=100):
        try:
            data = {'Time': np.sort(np.random.uniform(0, self._MAX_TRANSACTION_TIME, n_samples))}

            fraud_ranges = self.params['fraud']['amount_ranges']
            fraud_type = np.random.choice(
                ['small', 'medium', 'large'],
                size=n_samples,
                p=[fraud_ranges['small']['proportion'],
                   fraud_ranges['medium']['proportion'],
                   fraud_ranges['large']['proportion']]
            )

            amounts = []
            for ft in fraud_type:
                if ft == 'small':
                    amounts.append(np.random.uniform(
                        fraud_ranges['small']['min'],
                        fraud_ranges['small']['max']
                    ))
                elif ft == 'medium':
                    amounts.append(np.random.uniform(
                        fraud_ranges['medium']['min'],
                        fraud_ranges['medium']['max']
                    ))
                else:
                    amounts.append(np.random.uniform(
                        fraud_ranges['large']['min'],
                        fraud_ranges['large']['max']
                    ))

            data['Amount'] = np.array(amounts)

            for i in range(1, 29):
                v_col = f'V{i}'
                v_params = self.params['fraud']['v_features'][v_col]

                data[v_col] = np.random.normal(
                    v_params['mean'],
                    v_params['std'],
                    n_samples
                )

            data['Class'] = 1

            return pd.DataFrame(data)
        except Exception as e:
            logger.error("Unexpected error: ", e)
            raise e

    def generate_dataset(self, n_normal=10000, n_fraud=100, shuffle=True):
        try:
            normal_df = self.generate_normal_transactions(n_normal)
            fraud_df = self.generate_fraudulent_transactions(n_fraud)

            synthetic_data = pd.concat([normal_df, fraud_df], ignore_index=True)

            if shuffle:
                synthetic_data = synthetic_data.sample(frac=1, random_state=42).reset_index(drop=True)

            column_order = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount', 'Class']
            synthetic_data = synthetic_data[column_order]

            return synthetic_data
        except Exception as e:
            logger.error("Unexpected error: ", e)
            raise e

    def save_dataset(self, df, filename='synthetic_fraud_data.csv'):
        try:
            df.to_csv(filename, index=False)
            logger.info(f"Dataset saved to {filename}")
        except Exception as e:
            logger.error("Unexpected error: ", e)
            raise e