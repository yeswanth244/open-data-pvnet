import yaml

import pytest

from open_data_pvnet.utils.config_loader import load_config
from open_data_pvnet.utils.env_loader import load_environment_variables


def test_load_config_valid_yaml(tmp_path):
    # Create a temporary YAML file
    config_file = tmp_path / "config.yaml"
    config_content = """
    key1: value1
    key2:
        nested_key: value2
    """
    config_file.write_text(config_content)

    # Test loading the config
    config = load_config(str(config_file))

    assert isinstance(config, dict)
    assert config["key1"] == "value1"
    assert config["key2"]["nested_key"] == "value2"


def test_load_config_empty_path():
    with pytest.raises(ValueError, match="Missing config path"):
        load_config("")


def test_load_config_nonexistent_file():
    with pytest.raises(FileNotFoundError):
        load_config("nonexistent_file.yaml")


def test_load_config_invalid_yaml(tmp_path):
    # Create a temporary file with invalid YAML
    config_file = tmp_path / "invalid_config.yaml"
    config_file.write_text("key1: value1\n  invalid_indent: value2")

    with pytest.raises(yaml.YAMLError):
        load_config(str(config_file))


def test_load_environment_variables_success(tmp_path, monkeypatch):
    """Test successful loading of environment variables."""
    # Create a temporary .env file
    fake_env_content = "TEST_VAR=test_value\nANOTHER_VAR=123"
    env_file = tmp_path / ".env"
    env_file.write_text(fake_env_content)

    # Patch PROJECT_BASE to point to our temporary directory
    monkeypatch.setattr("open_data_pvnet.utils.env_loader.PROJECT_BASE", tmp_path)

    # Should not raise any exception
    load_environment_variables()


def test_load_environment_variables_file_not_found(tmp_path, monkeypatch):
    """Test that FileNotFoundError is raised when .env file doesn't exist."""
    # Patch PROJECT_BASE to point to our temporary directory
    monkeypatch.setattr("open_data_pvnet.utils.env_loader.PROJECT_BASE", tmp_path)

    # Should raise FileNotFoundError
    with pytest.raises(FileNotFoundError) as exc_info:
        load_environment_variables()

    assert ".env file not found" in str(exc_info.value)
