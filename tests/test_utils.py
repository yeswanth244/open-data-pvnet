from pathlib import Path

import pytest
import numpy as np
import xarray as xr
import yaml

from open_data_pvnet.utils.config_loader import load_config
from open_data_pvnet.utils.env_loader import load_environment_variables
from open_data_pvnet.utils.data_converters import convert_nc_to_zarr


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


@pytest.fixture
def temp_dirs(tmp_path):
    """Create temporary input and output directories."""
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    return input_dir, output_dir


@pytest.fixture
def sample_nc_file(temp_dirs):
    """Create a sample NetCDF file for testing."""
    input_dir, _ = temp_dirs
    nc_file = input_dir / "test_data.nc"

    # Create a simple dataset
    data = np.random.rand(10, 10)
    ds = xr.Dataset(
        {
            "temperature": (["x", "y"], data),
            "pressure": (["x", "y"], data * 2),
        },
        coords={
            "x": np.arange(10),
            "y": np.arange(10),
        },
    )
    ds.to_netcdf(nc_file)
    return nc_file


def test_successful_conversion(temp_dirs, sample_nc_file):
    """Test successful conversion of NC file to Zarr."""
    input_dir, output_dir = temp_dirs

    # Run conversion
    num_files, total_size = convert_nc_to_zarr(input_dir, output_dir)

    # Assertions
    assert num_files == 1
    assert total_size > 0
    assert (output_dir / "test_data.zarr").exists()

    # Verify data integrity
    original_ds = xr.open_dataset(sample_nc_file)
    converted_ds = xr.open_zarr(output_dir / "test_data.zarr")
    xr.testing.assert_equal(original_ds, converted_ds)

    original_ds.close()
    converted_ds.close()


def test_no_nc_files(temp_dirs):
    """Test behavior when no NC files are present."""
    input_dir, output_dir = temp_dirs

    num_files, total_size = convert_nc_to_zarr(input_dir, output_dir)

    assert num_files == 0
    assert total_size == 0


def test_overwrite_existing(temp_dirs, sample_nc_file):
    """Test overwrite behavior."""
    input_dir, output_dir = temp_dirs

    # First conversion
    convert_nc_to_zarr(input_dir, output_dir)

    # Second conversion without overwrite
    num_files, _ = convert_nc_to_zarr(input_dir, output_dir, overwrite=False)
    assert num_files == 0  # No files should be converted

    # Second conversion with overwrite
    num_files, _ = convert_nc_to_zarr(input_dir, output_dir, overwrite=True)
    assert num_files == 1  # File should be converted again


def test_invalid_input_dir():
    """Test behavior with invalid input directory."""
    with pytest.raises(Exception):
        convert_nc_to_zarr(Path("nonexistent_dir"), Path("output_dir"))
