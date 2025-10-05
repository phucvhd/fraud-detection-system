import pandas as pd

from services.generators.fraud_synthetic_generator import FraudSyntheticDataGenerator

mock_amount_std = 1.2
mock_amount_mean = 3.5
mock_features_std = {col: 1.5 for col in [f'V{i}' for i in range(1, 29)]}
mock_features_mean = {col: 0 for col in [f'V{i}' for i in range(1, 29)]}

test_df = pd.read_csv("test_creditcard.csv")
fraud_generator = FraudSyntheticDataGenerator(test_df)

def test_generate_normal_transactions():
    n_samples = 10
    df = fraud_generator.generate_normal_transactions(n_samples)
    assert not df.empty
    assert len(df) == n_samples