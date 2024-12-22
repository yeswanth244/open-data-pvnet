import argparse
import logging
from open_data_pvnet.scripts.archive import handle_archive
from open_data_pvnet.utils.env_loader import load_environment_variables

logger = logging.getLogger(__name__)

PROVIDERS = ["metoffice", "gfs", "dwd"]


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
    """Add a subparser for a specific data provider.

    Args:
        subparsers (argparse._SubParsersAction): The subparser container to add to
        provider_name (str): Name of the data provider (e.g., 'metoffice', 'gfs', 'dwd')

    Returns:
        argparse.ArgumentParser: The configured subparser for this provider
    """
    parser = subparsers.add_parser(
        provider_name, help=f"Commands for {provider_name.capitalize()} data"
    )
    parser.add_argument("operation", choices=["archive"], help="Operation to perform")
    parser.add_argument("--year", type=int, required=True, help="Year of data")
    parser.add_argument("--month", type=int, required=True, help="Month of data")


def configure_parser():
    """Configure the main argument parser for the CLI tool.

    Creates the main parser and adds subparsers for each supported data provider
    (metoffice, gfs, dwd). Each provider subparser includes options for year,
    month, and operation type.

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

    Initializes the environment, sets up command-line argument parsing,
    and handles the execution of the requested command. If no command
    is provided, displays the help message.
    """
    load_env_and_setup_logger()
    parser = configure_parser()
    args = parser.parse_args()

    # Handle the --list option
    if args.list == "providers":
        print("Available providers:")
        for provider in PROVIDERS:
            print(f"- {provider}")
        return

    # Dispatch based on the command
    if args.command:
        handle_archive(args.command, args.year, args.month)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
