"""

Written on Friday Apr 11th, 2025 in order to stream upload the large (160+ or 300+ GB) data files from a URL to an S3 bucket I have control over

"""

import requests
import boto3
from boto3.s3.transfer import TransferConfig
from tqdm import tqdm
import os

# AWS client
s3_client = boto3.client("s3")
bucket_name = "plantclef2025"
object_name = "PlantCLEF2024singleplanttrainingdata_800_max_side_size.tar"
file_path = "/tmp/largefile.tar"  # Temporary local path for the file

# Configure multipart upload
config = TransferConfig(
    multipart_threshold=1024 * 25,
    max_concurrency=10,
    multipart_chunksize=1024 * 25,
    use_threads=True,
)

# Stream the file directly from URL and upload to S3
url = "https://lab.plantnet.org/LifeCLEF/PlantCLEF2024/single_plant_training_data/PlantCLEF2024singleplanttrainingdata_800_max_side_size.tar"


##################################


def download_from_url(url: str, file_path: str, chunk_size: int = 1024 * 1024) -> bool:
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


##################################################################

if __name__ == "__main__":
    # Step 1: Download the file from the URL
    if download_from_url(url, file_path):
        print(f"File downloaded successfully to {file_path}.")
    else:
        print("Failed to download the file.")
        exit(1)

    # Step 2: Upload the file to S3
    if upload_local_file_to_s3(file_path, bucket_name, object_name, config):
        print(f"File uploaded successfully to s3://{bucket_name}/{object_name}.")
    else:
        print("Failed to upload the file.")
        exit(1)

# Clean up the temporary file (optional)
os.remove(file_path)
print(f"Temporary file {file_path} removed.")
print("File uploaded successfully!")
