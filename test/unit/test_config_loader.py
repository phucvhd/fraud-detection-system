from unittest.mock import mock_open, patch

import pytest

from config.config_loader import ConfigLoader


@patch("config.config_loader.load_dotenv")
@patch("os.path.exists", return_value=True)
def test_substitutes_multiple_vars_embedded_in_one_string(mock_exists, mock_load_dotenv, monkeypatch):
    # A single string value can contain several ${VAR} references (e.g. a DB
    # URL) — not just be one ${VAR} on its own.
    monkeypatch.setenv("POSTGRES_USER", "user")
    monkeypatch.setenv("POSTGRES_PASSWORD", "pass")
    monkeypatch.setenv("POSTGRES_DB", "db")

    yaml_content = (
        'database:\n  url: "postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@localhost:5432/${POSTGRES_DB}"\n'
    )
    with patch("builtins.open", mock_open(read_data=yaml_content)):
        loader = ConfigLoader()

    assert loader.config["database"]["url"] == "postgresql://user:pass@localhost:5432/db"


@patch("config.config_loader.load_dotenv")
@patch("os.path.exists", return_value=True)
def test_default_value_syntax_still_works(mock_exists, mock_load_dotenv, monkeypatch):
    monkeypatch.delenv("KAFKA_BOOTSTRAP_SERVERS", raising=False)
    yaml_content = 'kafka:\n  bootstrap_servers: "${KAFKA_BOOTSTRAP_SERVERS:-localhost:9092}"\n'
    with patch("builtins.open", mock_open(read_data=yaml_content)):
        loader = ConfigLoader()

    assert loader.config["kafka"]["bootstrap_servers"] == "localhost:9092"


@patch("config.config_loader.load_dotenv")
@patch("os.path.exists", return_value=True)
def test_default_value_syntax_prefers_actual_env_value(mock_exists, mock_load_dotenv, monkeypatch):
    monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:29092")
    yaml_content = 'kafka:\n  bootstrap_servers: "${KAFKA_BOOTSTRAP_SERVERS:-localhost:9092}"\n'
    with patch("builtins.open", mock_open(read_data=yaml_content)):
        loader = ConfigLoader()

    assert loader.config["kafka"]["bootstrap_servers"] == "kafka:29092"


@patch("config.config_loader.load_dotenv")
@patch("os.path.exists", return_value=True)
def test_missing_required_var_raises(mock_exists, mock_load_dotenv, monkeypatch):
    monkeypatch.delenv("SOME_REQUIRED_VAR", raising=False)
    yaml_content = 'key: "${SOME_REQUIRED_VAR}"\n'
    with patch("builtins.open", mock_open(read_data=yaml_content)):
        with pytest.raises(ValueError):
            ConfigLoader()
