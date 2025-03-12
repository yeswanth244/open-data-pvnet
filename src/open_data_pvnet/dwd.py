import xarray as xr
import requests
import zarr
import os
from pathlib import Path
import logging
from open_data_pvnet.utils.data_uploader import upload_to_huggingface
from open_data_pvnet.utils.config_loader import load_config

logger = logging.getLogger(__name__)

# Load DWD config
CONFIG_PATH = Path("configs/dwd_data_config.yaml")
CONFIG = load_config(CONFIG_PATH)

DWD_DATA_URL = CONFIG["input_data"]["nwp"]["dwd"]["source_url"]
ZARR_OUTPUT_DIR = Path(CONFIG["input_data"]["nwp"]["dwd"]["zarr_path"])

def process_dwd_data(year: int, month: int, day: int, overwrite: bool = False):
    """Download, process, and upload DWD weather data."""
    # Construct URL for the data file
    file_name = f"ICON_EU_{year}{month:02d}{day:02d}.nc"
    url = f"{DWD_DATA_URL}/{year}/{month:02d}/{day:02d}/{file_name}"

    local_file = Path(f"data/dwd/{file_name}")
    zarr_path = Path(f"data/dwd/dwd_data_{year}_{month:02d}_{day:02d}.zarr.zip")

    # Step 1: Download the data
    logger.info(f"Downloading DWD data from {url}")
    response = requests.get(url, stream=True)
    if response.status_code != 200:
        logger.error(f"Failed to download DWD data: {response.status_code}")
        return
    
    # Save the file locally
    local_file.parent.mkdir(parents=True, exist_ok=True)
    with open(local_file, "wb") as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)

    logger.info(f"Saved raw DWD data to {local_file}")

    # Step 2: Convert to .zarr.zip format
    logger.info(f"Converting {local_file} to Zarr format")
    ds = xr.open_dataset(local_file)

    # Restructure dataset before saving
    ds = restructure_dwd_dataset(ds)

    # Save as Zarr
    ds.to_zarr(zarr_path, mode="w", consolidated=True)
    logger.info(f"Saved Zarr file to {zarr_path}")

    # Step 3: Upload to Hugging Face
    logger.info(f"Uploading {zarr_path} to Hugging Face")
    upload_to_huggingface(
        config_path=CONFIG_PATH,
        folder_name=f"dwd_data_{year}_{month:02d}_{day:02d}",
        year=year,
        month=month,
        day=day,
        overwrite=overwrite,
        archive_type="zarr.zip"
    )

    logger.info("DWD data processing completed successfully.")


    def restructure_dwd_dataset(ds: xr.Dataset) -> xr.Dataset:
    """
    Restructure the dataset to match the required format.

    - Convert forecast period into "step" dimension.
    - Ensure all required metadata and coordinates are correct.
    - Drop unnecessary variables.

    Args:
        ds (xr.Dataset): Original dataset.

    Returns:
        xr.Dataset: Restructured dataset.
    """
    # Convert forecast_period and reference_time into dimensions
    if "forecast_period" in ds.coords and "forecast_period" not in ds.dims:
        ds = ds.expand_dims("forecast_period")

    # Rename dimensions to be consistent with Met Office data
    dim_mapping = {"forecast_period": "step", "forecast_reference_time": "initialization_time"}
    ds = ds.rename(dim_mapping)

    # Remove unnecessary coordinates
    for var in ["height", "bnds"]:
        if var in ds.coords:
            ds = ds.drop_vars(var)

    logger.info("Dataset successfully restructured.")
    return ds
