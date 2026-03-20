from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from src.generators.fraud_synthetic_generator import FraudSyntheticDataGenerator


@pytest.fixture
def mock_config_loader():
    config = Mock()
    config.config = {
        "fraud_generator": {
            "max_transaction_time": 3600,
            "fraud_rate": 0.01
        }
    }
    return config


@pytest.fixture
def mock_df():
    np.random.seed(42)
    normal_data = {
        "Time": np.random.uniform(0, 3600, 100),
        "Amount": np.random.uniform(1, 100, 100),
        "Class": [0] * 100
    }
    fraud_data = {
        "Time": np.random.uniform(0, 3600, 10),
        "Amount": np.random.uniform(10, 1000, 10),
        "Class": [1] * 10
    }
    for i in range(1, 29):
        normal_data[f"V{i}"] = np.random.normal(0, 1, 100)
        fraud_data[f"V{i}"] = np.random.normal(0, 2, 10)
        
    df = pd.concat([pd.DataFrame(normal_data), pd.DataFrame(fraud_data)]).reset_index(drop=True)
    return df


@pytest.fixture
def generator(mock_config_loader, mock_df):
    return FraudSyntheticDataGenerator(mock_config_loader, mock_df, seed=42)


def test_calibrate_from_dataframe(generator):
    assert "normal" in generator.params
    assert "fraud" in generator.params
    assert "v_features" in generator.params["normal"]
    assert "amount" in generator.params["normal"]
    assert "amount_ranges" in generator.params["fraud"]


def test_generate_normal_transaction(generator):
    tx = generator.generate_normal_transaction(100)
    assert tx["Time"] == 100
    assert tx["Class"] == 0
    assert tx["Amount"] > 0
    assert all(f"V{i}" in tx for i in range(1, 29))


def test_generate_fraudulent_transaction(generator):
    tx = generator.generate_fraudulent_transaction(150)
    assert tx["Time"] == 150
    assert tx["Class"] == 1
    assert tx["Amount"] > 0
    assert all(f"V{i}" in tx for i in range(1, 29))


def test_generate_normal_transactions(generator):
    df = generator.generate_normal_transactions(10)
    assert len(df) == 10
    assert all(df["Class"] == 0)


def test_generate_fraudulent_transactions(generator):
    df = generator.generate_fraudulent_transactions(5)
    assert len(df) == 5
    assert all(df["Class"] == 1)


def test_generate_transaction(generator):
    np.random.seed(42)
    tx_normal = generator.generate_transaction(200)
    assert "Amount" in tx_normal
    assert "Class" in tx_normal


def test_generate_dataset(generator):
    df = generator.generate_dataset(10, 2, shuffle=False)
    assert len(df) == 12
    assert sum(df["Class"] == 0) == 10
    assert sum(df["Class"] == 1) == 2
    
def test_generate_dataset_shuffle(generator):
    df = generator.generate_dataset(10, 2, shuffle=True)
    assert len(df) == 12
