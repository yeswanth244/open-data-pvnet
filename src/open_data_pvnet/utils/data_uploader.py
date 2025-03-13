import os
import logging
import tarfile
from pathlib import Path
from huggingface_hub import HfApi
from open_data_pvnet.utils.config_loader import load_config
import zarr
from zarr.storage import ZipStore

logger = logging.getLogger(__name__)

# Make functions importable in other files
__all__ = [
    "upload_to_huggingface",
    "upload_monthly_zarr",
    "upload_monthly_dwd",
    "_validate_token",
    "_upload_archive",
    "_validate_config",
    "create_tar_archive",
    "create_zarr_zip",
]


def _validate_config(config):
    """
    Validate the configuration and return required values.

    Args:
        config (dict): Configuration dictionary loaded from a YAML file.

    Returns:
        tuple: (repo_id, zarr_base_path)

    Raises:
        ValueError: If required configuration fields are missing.
    """
    repo_id = config.get("general", {}).get("destination_dataset_id")
    if not repo_id:
        raise ValueError("No destination_dataset_id found in the configuration file.")

    if "input_data" not in config or "nwp" not in config["input_data"]:
        raise ValueError("Missing 'input_data.nwp' section in the configuration file.")

    nwp_config = config["input_data"]["nwp"]

    # Handle DWD Configuration
    if "dwd" in nwp_config:
        local_output_dir = nwp_config["dwd"].get("local_output_dir")
        if not local_output_dir:
            raise ValueError("Missing 'local_output_dir' for DWD in the config file.")
        zarr_base_path = Path(local_output_dir) / "zarr"
        return repo_id, zarr_base_path

    # Handle Met Office Configuration
    if "met_office" in nwp_config:
        local_output_dir = nwp_config["met_office"].get("local_output_dir")
        if not local_output_dir:
            raise ValueError("Missing 'local_output_dir' for Met Office in the config file.")
        zarr_base_path = Path(local_output_dir) / "zarr"
        return repo_id, zarr_base_path

    raise ValueError("Configuration must contain either 'dwd' or 'met_office' under 'input_data.nwp'.")


def _validate_token():
    """
    Validate Hugging Face token and return API instance.

    Returns:
        tuple: (HfApi instance, authentication token)

    Raises:
        ValueError: If authentication fails.
    """
    hf_token = os.getenv("HUGGINGFACE_TOKEN")
    if not hf_token:
        raise ValueError("Hugging Face token not found. Set HUGGINGFACE_TOKEN in environment variables.")

    hf_api = HfApi()
    try:
        user_info = hf_api.whoami(token=hf_token)
        logger.info(f"Authenticated with Hugging Face as user: {user_info['name']}")
        return hf_api, hf_token
    except Exception as e:
        raise ValueError(f"Failed to authenticate with Hugging Face. Details: {e}")


def _ensure_repository(hf_api, repo_id, hf_token):
    """
    Ensure that a dataset repository exists on Hugging Face, create it if necessary.

    Args:
        hf_api (HfApi): Hugging Face API instance.
        repo_id (str): The dataset repository ID.
        hf_token (str): Hugging Face authentication token.
    """
    try:
        hf_api.dataset_info(repo_id, token=hf_token)
        logger.info(f"Found existing repository: {repo_id}")
    except Exception:
        logger.info(f"Creating new dataset repository: {repo_id}")
        hf_api.create_repo(repo_id=repo_id, repo_type="dataset", token=hf_token)


def _upload_archive(
    hf_api,
    archive_path: Path,
    repo_id: str,
    hf_token: str,
    overwrite: bool,
    year: int,
    month: int,
    day: int,
):
    """
    Upload an archive file to Hugging Face dataset repository.

    Args:
        hf_api (HfApi): The Hugging Face API instance.
        archive_path (Path): Path to the archive file.
        repo_id (str): Repository ID.
        hf_token (str): Hugging Face authentication token.
        overwrite (bool): Whether to overwrite existing files.
        year (int): Year for dataset structure.
        month (int): Month for dataset structure.
        day (int): Day for dataset structure.
    """
    target_path = f"data/{year:04d}/{month:02d}/{day:02d}/{archive_path.name}"
    logger.info(f"Uploading archive {archive_path} to {repo_id}:{target_path}")

    try:
        if overwrite:
            try:
                hf_api.delete_file(path_in_repo=target_path, repo_id=repo_id, repo_type="dataset", token=hf_token)
                logger.info(f"Deleted existing file {target_path} from repository")
            except Exception as e:
                logger.debug(f"File {target_path} not found in repository or couldn't be deleted: {e}")

        hf_api.upload_file(
            path_or_fileobj=str(archive_path),
            path_in_repo=target_path,
            repo_id=repo_id,
            repo_type="dataset",
            token=hf_token,
        )
        logger.info(f"Upload completed for {archive_path} to {repo_id}:{target_path}")
    except Exception as e:
        raise RuntimeError(f"Failed to upload archive: {e}")


def upload_to_huggingface(
    config_path: Path, folder_name: str, year: int, month: int, day: int, overwrite: bool = False, archive_type: str = "zarr.zip"
):
    """
    Upload a dataset folder to Hugging Face.

    Args:
        config_path (Path): Path to configuration file.
        folder_name (str): Name of the folder to upload.
        year (int): Year of data.
        month (int): Month of data.
        day (int): Day of data.
        overwrite (bool): Whether to overwrite existing files.
        archive_type (str): Type of archive to create ("zarr.zip" or "tar").
    """
    config = load_config(config_path)
    repo_id, zarr_base_path = _validate_config(config)

    hf_api, hf_token = _validate_token()
    _ensure_repository(hf_api, repo_id, hf_token)

    local_path = zarr_base_path / folder_name
    if not local_path.exists():
        raise FileNotFoundError(f"Zarr file not found: {local_path}")

   
    if archive_type == "zarr.zip":
        archive_path = create_zarr_zip(local_path, f"{folder_name}.zarr.zip", overwrite=overwrite)
    else:
        archive_path = create_tar_archive(local_path, f"{folder_name}.tar.gz", overwrite=overwrite)

    
    _upload_archive(hf_api, archive_path, repo_id, hf_token, overwrite, year, month, day)
    logger.info(f"Uploaded: {archive_path}")


    try:
        archive_path.unlink()
        logger.info(f"Deleted local archive: {archive_path}")
    except Exception as e:
        logger.warning(f"Failed to delete archive {archive_path}: {e}")


def upload_monthly_zarr(config_path: Path, year: int, month: int, overwrite: bool = False):
    """
    Upload a monthly consolidated Zarr dataset for Met Office data.

    Args:
        config_path (Path): Path to configuration file.
        year (int): Year of data.
        month (int): Month of data.
        overwrite (bool): Whether to overwrite existing files.
    """
    repo_id = "openclimatefix/met-office-uk-deterministic-solar"
    hf_api, hf_token = _validate_token()
    _ensure_repository(hf_api, repo_id, hf_token)

    monthly_file = f"met_office_{year}-{month:02d}.zarr.zip"
    local_path = Path("data/met_office/monthly") / monthly_file

    if not local_path.exists():
        raise FileNotFoundError(f"Monthly consolidated file not found: {local_path}")

    _upload_archive(hf_api, local_path, repo_id, hf_token, overwrite, year, month, None)
    logger.info(f"Uploaded monthly file: {local_path}")


def upload_monthly_dwd(config_path: Path, year: int, month: int, overwrite: bool = False):
    """
    Upload a monthly consolidated Zarr dataset for DWD data.

    Args:
        config_path (Path): Path to configuration file.
        year (int): Year of data.
        month (int): Month of data.
        overwrite (bool): Whether to overwrite existing files.
    """
    repo_id = "openclimatefix/dwd-weather-data"
    hf_api, hf_token = _validate_token()
    _ensure_repository(hf_api, repo_id, hf_token)

    monthly_file = f"dwd_data_{year}-{month:02d}.zarr.zip"
    local_path = Path("data/dwd/monthly") / monthly_file

    if not local_path.exists():
        raise FileNotFoundError(f"Monthly consolidated file not found: {local_path}")

    _upload_archive(hf_api, local_path, repo_id, hf_token, overwrite, year, month, None)
    logger.info(f"Uploaded monthly file: {local_path}")

def create_tar_archive(folder_path: Path, archive_name: str, overwrite: bool = False) -> Path:
    """
    Create a .tar.gz archive of the given folder.

    Args:
        folder_path (Path): The folder to archive.
        archive_name (str): Name of the archive file.
        overwrite (bool): Whether to overwrite an existing archive.

    Returns:
        Path: The path to the created archive.

    Raises:
        RuntimeError: If archive creation fails.
    """
    archive_path = folder_path.parent / archive_name

    if archive_path.exists():
        if overwrite:
            logger.info(f"Overwriting existing archive: {archive_path}")
            archive_path.unlink()
        else:
            logger.info(f"Archive already exists: {archive_path}. Skipping creation.")
            return archive_path

    try:
        with tarfile.open(archive_path, "w:gz") as tar:
            tar.add(folder_path, arcname=folder_path.name)
        logger.info(f"Created archive: {archive_path}")
    except Exception as e:
        raise RuntimeError(f"Failed to create archive: {e}")

    return archive_path

def create_zarr_zip(folder_path: Path, archive_name: str, overwrite: bool = False) -> Path:
    """
    Create a zip archive of a Zarr directory using zarr.zip functionality.

    Args:
        folder_path (Path): The Zarr folder to archive.
        archive_name (str): Name of the archive file (should end with .zip).
        overwrite (bool): Whether to overwrite the existing archive.

    Returns:
        Path: The path to the created archive.

    Raises:
        RuntimeError: If archive creation fails.
        ValueError: If the input folder is not a valid Zarr directory.
    """
    if not archive_name.endswith(".zip"):
        archive_name = f"{archive_name}.zip"

    archive_path = folder_path.parent / archive_name

    if archive_path.exists() and not overwrite:
        logger.info(f"Archive already exists: {archive_path}. Skipping creation.")
        return archive_path

    if archive_path.exists() and overwrite:
        logger.info(f"Overwriting existing archive: {archive_path}")
        archive_path.unlink()  # Delete the existing archive

    try:
        # Try to open the Zarr directory to verify it's valid
        try:
            zarr.open(str(folder_path))
        except Exception as e:
            raise ValueError(f"Not a valid Zarr directory: {folder_path}. Error: {e}")

        # Create zip archive
        logger.info(f"Creating Zarr zip archive: {archive_path}")

        # Open original Zarr directory
        source_store = zarr.DirectoryStore(str(folder_path))

        # Create new zip store and copy data
        with ZipStore(str(archive_path), mode="w") as zip_store:
            zarr.copy_store(source_store, zip_store)

        logger.info(f"Created Zarr zip archive: {archive_path}")
        return archive_path

    except Exception as e:
        if archive_path.exists():
            archive_path.unlink()  # Clean up partial archive on failure
        raise RuntimeError(f"Failed to create Zarr zip archive: {e}")
