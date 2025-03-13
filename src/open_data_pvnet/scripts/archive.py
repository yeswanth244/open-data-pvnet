import logging
from open_data_pvnet.nwp.met_office import process_met_office_data
from open_data_pvnet.nwp.gfs import process_gfs_data
from open_data_pvnet.nwp.dwd import process_dwd_data

logger = logging.getLogger(__name__)

def handle_archive(
    provider: str,
    year: int,
    month: int,
    day: int,
    hour: int = None,
    region: str = "global",
    overwrite: bool = False,
    archive_type: str = "zarr.zip",  #  Kept for flexibility
):
    """
    Handle the archive operation for different weather data providers.

    Args:
        provider (str): The data provider ('metoffice', 'gfs', 'dwd').
        year (int): Year of the data.
        month (int): Month of the data.
        day (int): Day of the data.
        hour (Optional[int]): Hour of the data (0-23). If None, processes all hours.
        region (str): Region for Met Office data ('global' or 'uk'). Default: 'global'.
        overwrite (bool): Whether to overwrite existing files. Default: False.
        archive_type (str): Type of archive to create ("zarr.zip" or "tar"). Default: "zarr.zip".

    Raises:
        NotImplementedError: If the provider is not recognized.
    """

    #  Handle Met Office Data
    if provider == "metoffice":
        if region not in ["global", "uk"]:
            raise ValueError(f"Invalid region '{region}'. Must be 'global' or 'uk'.")

        logger.info(f"Processing Met Office data for {year}-{month:02d}-{day:02d} with region={region}")
        handle_met_office_archive(
            year=year,
            month=month,
            day=day,
            hour=hour,
            region=region,
            overwrite=overwrite,
            archive_type=archive_type,
        )

    #  Handle GFS Data
    elif provider == "gfs":
        logger.info(f"Processing GFS data for {year}-{month:02d}-{day:02d} with overwrite={overwrite}")
        process_gfs_data(year, month, day, hour, overwrite=overwrite)

    # Handle DWD Data (Processes all hours if `hour=None`)
    elif provider == "dwd":
        hours = range(24) if hour is None else [hour]  # Process all hours if hour is None
        for hr in hours:
            logger.info(f"Processing DWD data for {year}-{month:02d}-{day:02d} at hour {hr:02d} with overwrite={overwrite}")
            process_dwd_data(
                year=year,
                month=month,
                day=day,
                hour=hr,
                overwrite=overwrite,
            )

    # Raise an error if provider is not recognized
    else:
        raise NotImplementedError(f"Provider '{provider}' is not yet implemented")