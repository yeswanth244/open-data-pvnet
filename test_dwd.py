import pytest
from pathlib import Path
from open_data_pvnet.nwp.dwd import process_dwd_data

def test_process_dwd_data():
    """Test if DWD processing correctly creates a .zarr.zip file."""
    process_dwd_data(year=2024, month=3, day=10, overwrite=True)
    assert Path("data/dwd/dwd_data_2024_03_10.zarr.zip").exists()
