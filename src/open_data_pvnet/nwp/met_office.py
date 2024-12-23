import logging
from pathlib import Path

from open_data_pvnet.utils.env_loader import PROJECT_BASE
from open_data_pvnet.utils.config_loader import load_config
from open_data_pvnet.utils.data_converters import convert_nc_to_zarr

import boto3

logger = logging.getLogger(__name__)

# Define the configuration paths for UK and Global
CONFIG_PATHS = {
    "uk": PROJECT_BASE / "src/open_data_pvnet/configs/met_office_uk_data_config.yaml",
    "global": PROJECT_BASE / "src/open_data_pvnet/configs/met_office_global_data_config.yaml",
}


def generate_prefix(region: str, year: int, month: int, day: int, hour: int) -> str:
    """
    Generate the S3 prefix for the given region, date, and hour.

    Args:
        region (str): Either 'uk' or 'global'.
        year (int): Year (e.g., 2022).
        month (int): Month (e.g., 12).
        day (int): Day (e.g., 1).
        hour (int): Hour in 24-hour format (e.g., 0 for 00:00Z).

    Returns:
        str: The S3 prefix for the specified date and hour.
    """
    folder = "uk-deterministic-2km" if region == "uk" else "global-deterministic-10km"
    return f"{folder}/{year:04d}{month:02d}{day:02d}T{hour:02d}00Z/"


def fetch_met_office_data(year: int, month: int, day: int, hour: int, region: str):
    """
    Fetch Met Office NWP data for the specified year, month, day, hour, and region.

    Args:
        year (int): Year to fetch data for.
        month (int): Month to fetch data for.
        day (int): Day to fetch data for.
        hour (int): Hour of data in 24-hour format.
        region (str): Region to fetch data for ("uk" or "global").
    """
    # Validate the region
    if region not in CONFIG_PATHS:
        raise ValueError(f"Invalid region '{region}'. Must be 'uk' or 'global'.")

    # Load configuration based on the region
    config_path = CONFIG_PATHS[region]
    config = load_config(config_path)

    logger.info(f"Loaded configuration for Met Office ({region})")
    logger.debug(f"Met Office Config ({region}): {config}")

    # Parse configuration for S3 bucket, paths, and required channels
    s3_bucket = config["input_data"]["nwp"]["met_office"]["s3_bucket"]
    local_output_dir = config["input_data"]["nwp"]["met_office"]["local_output_dir"]
    nwp_channels = set(config["input_data"]["nwp"]["met_office"]["nwp_channels"])
    nwp_accum_channels = set(config["input_data"]["nwp"]["met_office"]["nwp_accum_channels"])
    required_files = nwp_channels | nwp_accum_channels

    # Define raw file directory
    raw_dir = (
        Path(PROJECT_BASE) / local_output_dir / "raw" / f"{year}-{month:02d}-{day:02d}-{hour:02d}"
    )
    raw_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        f"Fetching Met Office {region} data for {year}-{month:02d}-{day:02d} at hour {hour:02d} from S3 bucket: {s3_bucket}"
    )
    logger.info(f"Raw file output directory: {raw_dir}")

    # Initialize S3 client
    s3 = boto3.client("s3")

    # Generate the S3 prefix for the specified region, date, and hour
    prefix = generate_prefix(region, year, month, day, hour)

    # Variables to track downloaded data
    total_files = 0
    total_size_bytes = 0

    # List files in the S3 bucket with the specified prefix
    try:
        response = s3.list_objects_v2(Bucket=s3_bucket, Prefix=prefix)
        if "Contents" not in response:
            logger.warning(f"No files found in S3 bucket '{s3_bucket}' with prefix '{prefix}'")
            return 0, 0  # Return no files and no size if nothing is fetched

        # Download only the required files based on channels
        for obj in response["Contents"]:
            s3_key = obj["Key"]
            file_name = s3_key.split("/")[-1]
            variable_name = file_name.split("-")[-1].replace(".nc", "")

            if variable_name not in required_files:
                logger.debug(f"Skipping file: {file_name} (not in required channels)")
                continue

            local_file_path = raw_dir / file_name
            logger.info(f"Downloading {s3_key} to {local_file_path}")
            s3.download_file(s3_bucket, s3_key, str(local_file_path))

            total_files += 1
            total_size_bytes += obj["Size"]

        logger.info(
            f"Completed downloading {total_files} files, {total_size_bytes / (1024 ** 2):.2f} MB"
        )

    except Exception as e:
        logger.error(f"Error fetching Met Office {region} data: {e}")
        raise

    return total_files, total_size_bytes


def process_met_office_data(year: int, month: int, day: int, hour: int, region: str):
    """
    Fetch and convert Met Office data to Zarr format.

    Args:
        year (int): Year of data.
        month (int): Month of data.
        day (int): Day of data.
        hour (int): Hour of data.
        region (str): Region ('uk' or 'global').
    """
    # Load configuration
    config_path = CONFIG_PATHS[region]
    config = load_config(config_path)
    local_output_dir = config["input_data"]["nwp"]["met_office"]["local_output_dir"]

    # Define directories
    raw_dir = (
        Path(PROJECT_BASE) / local_output_dir / "raw" / f"{year}-{month:02d}-{day:02d}-{hour:02d}"
    )
    zarr_dir = (
        Path(PROJECT_BASE) / local_output_dir / "zarr" / f"{year}-{month:02d}-{day:02d}-{hour:02d}"
    )
    zarr_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Fetch data
    logger.info(f"Starting data fetch for {region} region...")
    try:
        total_files, total_size = fetch_met_office_data(year, month, day, hour, region)
        if total_files == 0:
            logger.warning("No files were downloaded. Skipping conversion step.")
            return
        logger.info(f"Fetched {total_files} files ({total_size / (1024 ** 2):.2f} MB)")
    except Exception as e:
        logger.error(f"Error during data fetch: {e}")
        return

    # Step 2: Convert downloaded files to Zarr
    logger.info("Starting conversion to Zarr format...")
    try:
        converted_files, converted_size = convert_nc_to_zarr(raw_dir, zarr_dir)
        logger.info(f"Converted {converted_files} files to Zarr format ({converted_size:.2f} MB)")
    except Exception as e:
        logger.error(f"Error during Zarr conversion: {e}")
        return

    logger.info(
        f"Process completed for {region} data at {year}-{month:02d}-{day:02d} {hour:02d}:00"
    )
