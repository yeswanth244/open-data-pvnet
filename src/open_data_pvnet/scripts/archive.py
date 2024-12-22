from open_data_pvnet.nwp.met_office import fetch_met_office_data
from open_data_pvnet.nwp.gfs import fetch_gfs_data
from open_data_pvnet.nwp.dwd import fetch_dwd_data

import logging

logger = logging.getLogger(__name__)

PROVIDERS = {
    "metoffice": fetch_met_office_data,
    "gfs": fetch_gfs_data,
    "dwd": fetch_dwd_data,
}


def handle_archive(provider, year, month):
    if provider in PROVIDERS:
        logger.info(f"Fetching {provider.capitalize()} data for {year}-{month}")
        PROVIDERS[provider](year, month)
    else:
        raise ValueError(f"Unknown provider: {provider}")
