import os
import logging
from pathlib import Path
from huggingface_hub import HfApi
from open_data_pvnet.utils.config_loader import load_config

logger = logging.getLogger(__name__)


def _validate_config(config):
    """Validate configuration and return required values."""
    repo_id = config.get("general", {}).get("destination_dataset_id")
    if not repo_id:
        raise ValueError("No destination_dataset_id found in the configuration file.")

    local_output_dir = config["input_data"]["nwp"]["met_office"]["local_output_dir"]
    zarr_base_path = Path(local_output_dir) / "zarr"
    return repo_id, zarr_base_path


def _validate_token():
    """Validate Hugging Face token and return API instance."""
    hf_token = os.getenv("HUGGINGFACE_TOKEN")
    if not hf_token:
        raise ValueError(
            "Hugging Face token not found. Ensure HUGGINGFACE_TOKEN is set in the environment."
        )

    hf_api = HfApi()
    try:
        user_info = hf_api.whoami(token=hf_token)
        logger.info(f"Authenticated with Hugging Face as user: {user_info['name']}")
    except Exception as e:
        raise ValueError(
            f"Failed to authenticate with Hugging Face. Check your token. Details: {e}"
        )
    return hf_api, hf_token


def _ensure_repository(hf_api, repo_id, hf_token):
    """Ensure repository exists, create if it doesn't."""
    try:
        hf_api.dataset_info(repo_id, token=hf_token)
        logger.info(f"Found existing repository: {repo_id}")
    except Exception:
        logger.info(f"Creating new dataset repository: {repo_id}")
        hf_api.create_repo(repo_id=repo_id, repo_type="dataset", token=hf_token)


def _upload_files(hf_api, local_path, repo_id, hf_token, overwrite):
    """Upload all files from local path to Hugging Face repository."""
    if not local_path.exists():
        raise FileNotFoundError(f"Local path does not exist: {local_path}")

    if overwrite:
        # Delete existing files in the repository that match our upload path
        try:
            existing_files = hf_api.list_repo_files(repo_id=repo_id, repo_type="dataset")
            for file in existing_files:
                if file.startswith(local_path.name):
                    logger.info(f"Deleting existing file: {file}")
                    hf_api.delete_file(
                        path_in_repo=file, repo_id=repo_id, repo_type="dataset", token=hf_token
                    )
        except Exception as e:
            logger.warning(f"Error while cleaning existing files: {e}")

    # Upload new files
    for file in local_path.rglob("*"):
        if file.is_file():
            target_path = file.relative_to(local_path).as_posix()
            logger.info(f"Uploading {file} to {repo_id}:{target_path}")
            hf_api.upload_file(
                path_or_fileobj=str(file),
                path_in_repo=target_path,
                repo_id=repo_id,
                repo_type="dataset",
                token=hf_token,
            )


def upload_to_huggingface(config_path: Path, folder_name: str, overwrite: bool = False):
    """
    Upload a specific folder from the local Zarr directory to a Hugging Face dataset repository.

    Args:
        config_path (Path): Path to the configuration YAML file.
        folder_name (str): Name of the folder to upload (e.g., '2022-12-01-00').
        overwrite (bool): Whether to overwrite existing files in the repository.

    Raises:
        Exception: If the upload fails due to authentication, network, or other issues.
    """
    try:
        # Load and validate configuration
        config = load_config(config_path)
        repo_id, zarr_base_path = _validate_config(config)

        # Setup authentication
        hf_api, hf_token = _validate_token()

        # Ensure repository exists
        _ensure_repository(hf_api, repo_id, hf_token)

        # Upload files
        local_path = zarr_base_path / folder_name
        _upload_files(hf_api, local_path, repo_id, hf_token, overwrite)

        logger.info(f"Upload to Hugging Face completed: {repo_id}")

    except Exception as e:
        logger.error(f"Error uploading to Hugging Face: {e}")
        raise
