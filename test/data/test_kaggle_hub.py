import pytest

from data.kaggle_hub import KaggleHub

kaggle_hub = KaggleHub()

def test_load_dataset():
    try:
        actual = kaggle_hub.load_dateset("mlg-ulb/creditcardfraud", "creditcard.csv")
        assert actual is not None
        assert not actual.empty
    except Exception as e:
        pytest.fail(f"Unexpected error: {e}")
