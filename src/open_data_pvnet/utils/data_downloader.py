import zarr
from pathlib import Path
import xarray as xr
import logging
from typing import Union, Optional
from huggingface_hub import hf_hub_download

logger = logging.getLogger(__name__)

DATASET_REPO = "openclimatefix/met-office-uk-deterministic-solar"


def load_zarr_data(
    archive_path: Union[str, Path],
    chunks: Optional[dict] = None,
    download: bool = True,
    consolidated: bool = False,
) -> xr.Dataset:
    """
    Load a zarr dataset from a zip archive using xarray. If the file doesn't exist locally
    and download=True, it will attempt to download it from Hugging Face first.

    Args:
        archive_path (Union[str, Path]): Path to the .zarr.zip file
        chunks (Optional[dict]): Dictionary specifying chunk sizes for each dimension.
            Example: {'time': 24, 'latitude': 100, 'longitude': 100}
            If None, will use the original chunking from the zarr store.
        download (bool): Whether to download the file from Hugging Face if not found locally
        consolidated (bool): Whether to use consolidated metadata when opening the zarr store.
            Set to False to avoid the warning about non-consolidated metadata.

    Returns:
        xr.Dataset: The loaded dataset

    Raises:
        ValueError: If the file doesn't exist or isn't a .zarr.zip file
        RuntimeError: If there's an error loading the dataset
    """
    archive_path = Path(archive_path)
    local_path = Path("data") / archive_path  # Add data prefix for local storage

    if not local_path.exists() and download:
        try:
            # Convert local path to repository path (data/year/month/day/file.zarr.zip)
            repo_path = f"data/{str(archive_path)}"
            logger.info(f"Downloading {repo_path} from Hugging Face...")

            # Ensure parent directory exists
            local_path.parent.mkdir(parents=True, exist_ok=True)

            # Download the file from Hugging Face
            hf_hub_download(
                repo_id=DATASET_REPO,
                filename=repo_path,
                local_dir=".",  # Download to current directory
                repo_type="dataset",
            )
            logger.info(f"Successfully downloaded to {local_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to download dataset from Hugging Face: {e}")

    if not local_path.exists():
        raise ValueError(f"Archive not found: {local_path}")

    if not str(local_path).endswith(".zarr.zip"):
        raise ValueError(f"File must be a .zarr.zip archive: {local_path}")

    try:
        # Open the zip store
        store = zarr.storage.ZipStore(str(local_path), mode="r")

        # Load the dataset with optional chunking and explicit consolidated setting
        ds = xr.open_zarr(store, chunks=chunks, consolidated=consolidated)

        logger.info(f"Successfully loaded dataset from {local_path}")
        return ds

    except Exception as e:
        raise RuntimeError(f"Error loading zarr dataset: {e}")

    finally:
        # Make sure to close the store
        if "store" in locals():
            store.close()
