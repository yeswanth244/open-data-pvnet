import argparse
import logging
from open_data_pvnet.scripts.archive import handle_archive
from open_data_pvnet.utils.env_loader import load_environment_variables
from open_data_pvnet.utils.data_downloader import load_zarr_data
from pathlib import Path

logger = logging.getLogger(__name__)

PROVIDERS = ["metoffice", "gfs", "dwd"]
DEFAULT_REGION = "global"  # Default region for Met Office datasets


def load_env_and_setup_logger():
    """Initialize environment variables and configure logging.

    This function performs two main tasks:
    1. Loads environment variables from configuration files
    2. Sets up basic logging configuration with INFO level

    Raises:
        FileNotFoundError: If the environment configuration file cannot be found
    """
    try:
        load_environment_variables()
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
        logger.info("Environment variables loaded successfully.")
    except FileNotFoundError as e:
        logger.error(f"Error loading environment variables: {e}")
        raise


def add_provider_parser(subparsers, provider_name):
    """Add a subparser for a specific data provider."""
    provider_parser = subparsers.add_parser(
        provider_name, help=f"Commands for {provider_name.capitalize()} data"
    )
    operation_subparsers = provider_parser.add_subparsers(
        dest="operation", help="Operation to perform"
    )

    # Archive operation parser
    archive_parser = operation_subparsers.add_parser("archive", help="Archive data to Hugging Face")
    _add_common_arguments(archive_parser, provider_name)
    archive_parser.add_argument(
        "--archive-type",
        choices=["zarr.zip", "tar"],
        default="zarr.zip",
        help="Type of archive to create (default: zarr.zip)",
    )

    # Load operation parser
    load_parser = operation_subparsers.add_parser("load", help="Load archived data")
    _add_common_arguments(load_parser, provider_name)
    load_parser.add_argument(
        "--chunks",
        type=str,
        help="Chunking specification in format 'dim1:size1,dim2:size2' (e.g., 'time:24,latitude:100')",
    )


def _add_common_arguments(parser, provider_name):
    """Add arguments common to both archive and load operations."""
    parser.add_argument("--year", type=int, required=True, help="Year of data")
    parser.add_argument("--month", type=int, required=True, help="Month of data")
    parser.add_argument("--day", type=int, required=True, help="Day of data")

    # Add Met Office specific arguments
    if provider_name == "metoffice":
        parser.add_argument(
            "--hour",
            type=int,
            help="Hour of data (0-23). If not specified, process all hours of the day.",
            default=None,
        )
        parser.add_argument(
            "--region",
            choices=["global", "uk"],
            default=DEFAULT_REGION,
            help="Specify the Met Office dataset region (default: global)",
        )

    parser.add_argument(
        "--overwrite",
        "-o",
        action="store_true",
        help="Overwrite existing files in output directories",
    )


def parse_chunks(chunks_str):
    """Parse chunks string into dictionary."""
    if not chunks_str:
        return None
    chunks = {}
    for chunk in chunks_str.split(","):
        dim, size = chunk.split(":")
        chunks[dim.strip()] = int(size)
    return chunks


def handle_load(provider: str, year: int, month: int, day: int, **kwargs):
    """Handle loading archived data."""
    hour = kwargs.get("hour", 0)  # Default to hour 0 if not specified
    chunks = parse_chunks(kwargs.get("chunks"))

    # Construct the archive path based on provider and parameters
    # Format: data/2023/01/16/2023-01-16-00.zarr.zip
    archive_path = (
        Path("data")
        / str(year)
        / f"{month:02d}"
        / f"{day:02d}"
        / f"{year}-{month:02d}-{day:02d}-{hour:02d}.zarr.zip"
    )

    try:
        dataset = load_zarr_data(archive_path, chunks=chunks)
        logger.info(f"Successfully loaded dataset for {year}-{month:02d}-{day:02d} hour {hour:02d}")
        return dataset
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise


def configure_parser():
    """Configure the main argument parser for the CLI tool.

    Creates the main parser and adds subparsers for each supported data provider
    (metoffice, gfs, dwd). Each provider subparser includes options for year,
    month, day, hour, and operation type.

    Returns:
        argparse.ArgumentParser: The configured argument parser
    """
    parser = argparse.ArgumentParser(prog="open-data-pvnet", description="Open Data PVNet CLI Tool")
    parser.add_argument(
        "--list",
        choices=["providers"],
        help="List available options (e.g., providers)",
        action="store",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    for provider in PROVIDERS:
        add_provider_parser(subparsers, provider)
    return parser


def main():
    """Entry point for the Open Data PVNet CLI tool.

    For example:

    open-data-pvnet metoffice archive --year 2023 --month 12 --day 1 --region uk -o

    If you want to use a tar archive, you can do so with the following command:
    open-data-pvnet metoffice archive --year 2023 --month 12 --day 1 --hour 12 --region uk -o --archive-type tar
    """
    load_env_and_setup_logger()
    parser = configure_parser()
    args = parser.parse_args()

    if args.list == "providers":
        print("Available providers:")
        for provider in PROVIDERS:
            print(f"- {provider}")
        return

    if not args.command:
        parser.print_help()
        return

    kwargs = {
        "provider": args.command,
        "year": args.year,
        "month": args.month,
        "day": args.day,
        "hour": getattr(args, "hour", None),
        "overwrite": args.overwrite,
    }

    # Only add region for Met Office commands
    if args.command == "metoffice":
        kwargs["region"] = args.region

    if args.operation == "archive":
        kwargs["archive_type"] = args.archive_type
        handle_archive(**kwargs)
    elif args.operation == "load":
        kwargs["chunks"] = getattr(args, "chunks", None)
        handle_load(**kwargs)
    else:
        parser.print_help()
