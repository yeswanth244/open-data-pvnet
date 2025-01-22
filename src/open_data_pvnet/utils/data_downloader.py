import zarr
from pathlib import Path
import xarray as xr
import logging
from typing import Union, Optional, List
from huggingface_hub import hf_hub_download

logger = logging.getLogger(__name__)

DATASET_REPO = "openclimatefix/met-office-uk-deterministic-solar"


def download_from_hf(repo_path: str, local_path: Path) -> None:
    """Download a file from HuggingFace."""
    logger.info(f"Downloading {repo_path} from Hugging Face...")
    local_path.parent.mkdir(parents=True, exist_ok=True)
    hf_hub_download(
        repo_id=DATASET_REPO,
        filename=repo_path,
        local_dir=".",
        repo_type="dataset",
    )
    logger.info(f"Successfully downloaded to {local_path}")


def get_zarr_groups(store: zarr.storage.ZipStore) -> List[str]:
    """Get all top-level Zarr groups from the store."""
    return [k.split("/")[0] for k in store.keys() if k.endswith(".zarr/.zgroup")]


def open_zarr_group(
    store: zarr.storage.ZipStore, group: str, chunks: Optional[dict], consolidated: bool
) -> xr.Dataset:
    """Open a single Zarr group as an xarray Dataset."""
    return xr.open_zarr(store, group=group, chunks=chunks, consolidated=consolidated)


def merge_datasets(datasets: List[xr.Dataset]) -> xr.Dataset:
    """Merge multiple datasets into one and log the result."""
    ds = xr.merge(datasets, compat="override")
    logger.info("Dataset info:")
    logger.info(f"Variables: {list(ds.variables)}")
    logger.info(f"Dimensions: {dict(ds.dims)}")
    logger.info(f"Coordinates: {list(ds.coords)}")
    return ds


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
    store = None
    try:
        archive_path = Path(archive_path)
        local_path = archive_path

        if not local_path.exists() and download:
            download_from_hf(str(archive_path), local_path)

        logger.info(f"Opening zarr store from {local_path}")
        logger.info(f"File size: {local_path.stat().st_size / (1024*1024):.2f} MB")

        store = zarr.storage.ZipStore(str(local_path), mode="r")
        zarr_groups = get_zarr_groups(store)

        datasets = []
        for group in zarr_groups:
            try:
                group_ds = open_zarr_group(store, group, chunks, consolidated)
                datasets.append(group_ds)
            except Exception as e:
                logger.warning(f"Could not open group {group}: {e}")
                continue

        if datasets:
            return merge_datasets(datasets)
        else:
            raise ValueError("No valid datasets found in the Zarr store")

    except Exception as e:
        logger.error(f"Error loading zarr dataset: {e}")
        raise

    finally:
        if store is not None:
            store.close()
