"""
File: utils.py
Author: Jacob Alexander Rose


Written on  Sunday Apr 13th, 2025, containing some functions previously written on Friday Apr 11th, 2025 in order to
1. Download the full plantclef 2025 dataset from a 3rd party URL to the local dev environment
2. Stream upload the large (160+ or 300+ GB) data files from the local dev environment to an S3 bucket I have control over

"""

import requests
import boto3
from boto3.s3.transfer import TransferConfig
from tqdm import tqdm
import os
from typing import List, Union, Dict
import pandas as pd
from pathlib import Path
from functools import lru_cache
import shutil

# plantclef_class_index_path = Path(train_metadata_dir, "species_ids.csv")
# dataset_dir = "/teamspace/studios/this_studio/plantclef-vision/data/plantclef2025/PlantCLEF2024singleplanttrainingdata_800_max_side_size/images_max_side_800"


def optimize_pandas_dtypes(data_df: pd.DataFrame) -> pd.DataFrame:
    """
    Optimize the dtypes of a pandas DataFrame by converting to the most
    memory-efficient dtypes.

    """
    data_df = data_df.convert_dtypes()
    val_counts = data_df.nunique()

    max_categories = 10_000
    categorical_cols = val_counts[val_counts < max_categories].index.tolist()

    data_df = data_df.astype({col: "category" for col in categorical_cols})

    return data_df


def collect_image_filepaths(dataset_dir) -> List[str]:
    """
    Collect all image file paths from the dataset directory into a list of strings.
    Args:
        dataset_dir (str): Directory containing the image files.
    Returns:
        list: List of image file paths.
    """
    return [
        os.path.join(root, filename)
        for root, _, filenames in tqdm(
            os.walk(dataset_dir),
            smoothing=0,
            position=0,
            desc=f"Walking through dir {dataset_dir}",
        )
        for filename in tqdm(
            filenames,
            total=len(filenames),
            smoothing=0,
            position=1,
            desc=f"Collecting file paths in {root}",
        )
    ]


def merge_filepaths_with_metadata(metadata_df, dataset_dir) -> pd.DataFrame:
    """
    Merge the image file paths with the metadata DataFrame.
    Args:
        metadata_df (pd.DataFrame): DataFrame containing metadata.
        dataset_dir (str): Directory containing the image files.
    Returns:
        pd.DataFrame: Merged DataFrame containing metadata and image file paths.
    """
    image_paths = collect_image_filepaths(dataset_dir)
    paths_df = pd.DataFrame(image_paths, columns=["image_path"])
    paths_df = paths_df.assign(
        image_name=paths_df.image_path.map(lambda x: x.split("/")[-1])
    )

    return metadata_df.merge(paths_df, how="inner", on="image_name")


@lru_cache(maxsize=1)
def load_plantclef_idx2class(
    plantclef_class_index_path: Union[str, Path],
) -> Dict[int, int]:
    """
    Load the plantclef class index from the given path.
    The class index is a mapping of class IDs to their respective class names.
    Args:
        plantclef_class_index_path (Union[str, Path]): The path to the class index file.
    Returns:
        Dict[int, int]: A dictionary mapping class index # to their respective species ID.
    """
    class_mappings = pd.read_csv(plantclef_class_index_path)
    assert (
        (class_mappings.index.is_monotonic_increasing)
        and (class_mappings.index.is_unique)
    ), "Class index loaded from 'species_ids.csv' is not monotonic increasing or not unique. Please check the input file."

    idx2class = class_mappings.to_dict()["species_id"]
    return idx2class


@lru_cache(maxsize=1)
def load_plantclef_class2idx(
    plantclef_class_index_path: Union[str, Path],
) -> Dict[int, int]:
    """
    Load the plantclef class index from the given path.
    The class index is a mapping of class names to their respective IDs.
    Args:
        plantclef_class_index_path (Union[str, Path]): The path to the class index file.
    Returns:
        Dict[int, int]: A dictionary mapping class index # to their respective species ID.
    """
    class_mappings = pd.read_csv(plantclef_class_index_path)
    assert (
        (class_mappings.index.is_monotonic_increasing)
        and (class_mappings.index.is_unique)
    ), "Class index loaded from 'species_ids.csv' is not monotonic increasing or not unique. Please check the input file."

    class2idx = class_mappings.reset_index().set_index("species_id").to_dict()["index"]

    return class2idx


def download_from_url(
    url: str, file_path: str, chunk_size: int = 5 * 1024 * 1024
) -> bool:
    response = requests.get(url, stream=True)
    total_size = int(
        response.headers.get("Content-Length", 0)
    )  # Get file size in bytes

    if os.path.exists(file_path):
        if os.stat(file_path).st_size == total_size:
            print("File already downloaded completely, skipping to upload step.")
            return True
        else:
            print("Detected incomplete download from previous run, overwriting.")

    with (
        open(file_path, "wb") as f,
        tqdm(
            desc="Downloading file",
            total=total_size,  # Total size of the file
            unit="B",  # Bytes
            unit_scale=True,  # Automatically scale units (KB, MB, GB)
        ) as bar,
    ):
        for chunk in response.iter_content(
            chunk_size=chunk_size
        ):  # Adjust chunk size as needed
            if chunk:
                f.write(chunk)
                bar.update(len(chunk))
    try:
        assert os.stat(file_path).st_size == total_size
    except Exception:
        print("Downloaded file size does not match expected size.")
        return False
    return True


def upload_local_file_to_s3(
    file_path: str, bucket_name: str, object_name: str, config: TransferConfig
) -> bool:
    """
    Upload a local file to an S3 bucket.
    """
    total_size = os.stat(file_path).st_size
    try:
        s3_client = boto3.client("s3")
        # Step 2: Upload the file to S3 with a progress bar
        with tqdm(
            desc="Uploading to S3",
            total=total_size,  # Total size of the file
            unit="B",  # Bytes
            unit_scale=True,  # Automatically scale units (KB, MB, GB)
        ) as bar:
            s3_client.upload_file(
                file_path,
                bucket_name,
                object_name,
                Config=config,
                Callback=lambda bytes_transferred: bar.update(bytes_transferred),
            )
        return True
    except Exception as e:
        print(f"Error uploading file to S3: {e}")
        return False


def clear_cache(cache_dir: str) -> None:
    """Clear the cache directory."""
    try:
        if os.path.isdir(cache_dir):
            shutil.rmtree(cache_dir)
    except Exception:
        # print(f"Error clearing cache: {e}")
        pass
