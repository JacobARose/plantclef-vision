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
