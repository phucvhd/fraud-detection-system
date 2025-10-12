import numpy as np
import pandas as pd

from config.config_loader import ConfigLoader
from services.generators.fraud_synthetic_generator import FraudSyntheticDataGenerator

mock_amount_std = 1.2
mock_amount_mean = 3.5
mock_features_std = {col: 1.5 for col in [f'V{i}' for i in range(1, 29)]}
mock_features_mean = {col: 0 for col in [f'V{i}' for i in range(1, 29)]}

test_config = ConfigLoader(path="application-test.yaml")
test_df = pd.read_csv("test_creditcard.csv")
fraud_generator = FraudSyntheticDataGenerator(config_loader=test_config, df=test_df)

def test_generate_normal_transactions():
    n_samples = 10
    df = fraud_generator.generate_normal_transactions(n_samples)
    assert not df.empty
    assert len(df) == n_samples

def test_generate_transaction():
    n_samples = 10
    time_intervals = np.linspace(0, 100000, num=n_samples).tolist()
    transactions = []
    for time_interval in time_intervals:
        transaction = fraud_generator.generate_transaction(time_interval)
        transactions.append(transaction)
    df = pd.DataFrame(transactions)
    assert len(transactions) == n_samples
    assert len(df) == n_samples
