import numpy as np
import pandas as pd

from data.validator.validator import Validator

validator = Validator()

columns = ['Time'] + [f"V{i}" for i in range(1, 29)] + ['Amount', 'Class']
mock_data = [
    [0.0] + list(np.random.normal(0, 1, 28)) + [149.62, 0],
    [1.0] + list(np.random.normal(0, 1, 28)) + [2.69, 0],
    [2.0] + list(np.random.normal(0, 1, 28)) + [378.66, 1],
    [3.0] + list(np.random.normal(0, 1, 28)) + [123.50, 0],
    [4.0] + list(np.random.normal(0, 1, 28)) + [69.99, 1]
]

def test_validate_consistency():
    mock_dataframe = pd.DataFrame(mock_data, columns=columns)
    result = validator.validate_consistency(mock_dataframe)
    assert result is True

def test_validate_consistency_with_incorrect_shape():
    invalid_columns = ['Time'] + [f"V{i}" for i in range(1, 28)] + ['Amount', 'Class']
    invalid_mock_data = [
        [0.0] + list(np.random.normal(0, 1, 27)) + [149.62, 0],
        [1.0] + list(np.random.normal(0, 1, 27)) + [2.69, 0],
        [2.0] + list(np.random.normal(0, 1, 27)) + [378.66, 1],
        [3.0] + list(np.random.normal(0, 1, 27)) + [123.50, 0],
        [4.0] + list(np.random.normal(0, 1, 27)) + [69.99, 1]
    ]
    mock_dataframe = pd.DataFrame(invalid_mock_data, columns=invalid_columns)
    result = validator.validate_consistency(mock_dataframe)
    assert result is False

def test_validate_consistency_with_empty_data():
    empty_mock_data = []
    mock_dataframe = pd.DataFrame(empty_mock_data)
    result = validator.validate_consistency(mock_dataframe)
    assert result is False

def test_validate_consistency_with_non_numerical_type():
    invalid_mock_data = [
        [0.0] + list(np.random.normal(0, 1, 28)) + ["John", "Doe"]
    ]
    mock_dataframe = pd.DataFrame(invalid_mock_data, columns=columns)
    result = validator.validate_consistency(mock_dataframe)
    assert result is False

def test_validate_integrity():
    mock_dataframe = pd.DataFrame(mock_data, columns=columns)
    result = validator.validate_integrity(mock_dataframe)
    assert result is True

def test_validate_integrity_with_missing_values():
    invalid_mock_data = [
        [0.0] + list(np.random.normal(0, 1, 28)) + [np.nan, 0],
        [1.0] + list(np.random.normal(0, 1, 28)) + [2.69, 0],
        [2.0] + list(np.random.normal(0, 1, 28)) + [np.nan, 1],
        [3.0] + list(np.random.normal(0, 1, 28)) + [123.50, 0],
        [4.0] + list(np.random.normal(0, 1, 28)) + [69.99, np.nan]
    ]
    mock_dataframe = pd.DataFrame(invalid_mock_data, columns=columns)
    result = validator.validate_integrity(mock_dataframe)
    assert result is False

def test_fill_column_missing_values():
    invalid_mock_data = [
        [0.0] + list(np.random.normal(0, 1, 28)) + [np.nan, 0],
        [1.0] + list(np.random.normal(0, 1, 28)) + [2.69, 0],
        [2.0] + list(np.random.normal(0, 1, 28)) + [np.nan, 1],
        [3.0] + list(np.random.normal(0, 1, 28)) + [123.50, 0],
        [4.0] + list(np.random.normal(0, 1, 28)) + [69.99, 1]
    ]
    mock_dataframe = pd.DataFrame(invalid_mock_data, columns=columns)
    result = validator.fill_column_missing_values(mock_dataframe, "Amount")
    assert result is True
    assert mock_dataframe.isnull().sum().sum() == 0
