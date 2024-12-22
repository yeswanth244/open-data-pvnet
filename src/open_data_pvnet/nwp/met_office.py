import logging
from pathlib import Path

from open_data_pvnet.utils.env_loader import PROJECT_BASE
from open_data_pvnet.utils.config_loader import load_config

import boto3

logger = logging.getLogger(__name__)

# Load configuration for Met Office
CONFIG_PATH = PROJECT_BASE / "src/open_data_pvnet/configs/met_office_data_config.yaml"
CONFIG = load_config(CONFIG_PATH)

# Log configuration details for debugging
logger.info("Loaded configuration for Met Office")
logger.debug(f"Met Office Config: {CONFIG}")


def fetch_met_office_data(year: int, month: int):
    """
    Fetch Met Office NWP data for the specified year and month.

    Args:
        year (int): Year to fetch data for.
        month (int): Month to fetch data for.
    """
    # Parse configuration for S3 bucket and paths
    s3_bucket = CONFIG["input_data"]["nwp"]["met_office"]["s3_bucket"]
    s3_prefix = CONFIG["input_data"]["nwp"]["met_office"]["s3_prefix"]
    local_output_dir = CONFIG["input_data"]["nwp"]["met_office"]["local_output_dir"]

    logger.info(f"Fetching Met Office data for {year}-{month} from S3 bucket: {s3_bucket}")

    # Create the output directory if it doesn't exist
    output_path = Path(PROJECT_BASE) / local_output_dir / f"{year}-{month:02d}"
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Local output directory: {output_path}")

    # Initialize S3 client
    s3 = boto3.client("s3")

    # S3 prefix for the specified year and month
    prefix = f"{s3_prefix}/{year}/{month:02d}/"

    # List files in the S3 bucket with the specified prefix
    try:
        response = s3.list_objects_v2(Bucket=s3_bucket, Prefix=prefix)
        if "Contents" not in response:
            logger.warning(f"No files found in S3 bucket '{s3_bucket}' with prefix '{prefix}'")
            return

        # Download each file in the prefix
        for obj in response["Contents"]:
            s3_key = obj["Key"]
            file_name = s3_key.split("/")[-1]
            local_file_path = output_path / file_name

            logger.info(f"Downloading {s3_key} to {local_file_path}")
            s3.download_file(s3_bucket, s3_key, str(local_file_path))

        logger.info(f"Completed downloading Met Office data for {year}-{month}")

    except Exception as e:
        logger.error(f"Error fetching Met Office data: {e}")
        raise
