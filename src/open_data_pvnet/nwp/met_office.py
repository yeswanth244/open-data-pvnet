import logging

from open_data_pvnet.utils.env_loader import PROJECT_BASE
from open_data_pvnet.utils.config_loader import load_config

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
    logger.info(f"Fetching Met Office data for {year}-{month}")
    # Replace this with actual data fetching logic
    logger.warning("fetch_met_office_data is not yet implemented")
    # Replace this with actual data fetching logic
