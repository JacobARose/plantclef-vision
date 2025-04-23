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
from typing import List
import pandas as pd

# dataset_dir = "/teamspace/studios/this_studio/plantclef-vision/data/plantclef2025/PlantCLEF2024singleplanttrainingdata_800_max_side_size/images_max_side_800"


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
        for root, _, filenames in tqdm(os.walk(dataset_dir), smoothing=0)
        for filename in filenames
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
