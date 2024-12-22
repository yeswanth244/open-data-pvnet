import logging

logger = logging.getLogger(__name__)


def fetch_dwd_data(year, month):
    logger.info(f"Downloading DWD data for {year}-{month}")
    raise NotImplementedError("The fetch_dwd_data function is not implemented yet.")
