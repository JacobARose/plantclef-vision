"""
File: download_plantclef2025.py
Author: Jacob Alexander Rose

Written on Friday Apr 11th, 2025 in order to:
Download the full plantclef 2025 dataset from a 3rd party URL to the local dev environment

"""

from plantclef.datasets.utils import download_from_url

####################################################################

# Configure multipart upload
# config = TransferConfig(
#     multipart_threshold=1024 * 25,
#     max_concurrency=10,
#     multipart_chunksize=1024 * 25,
#     use_threads=True,
# )

##################################################################

if __name__ == "__main__":
    # Stream the file directly from URL and upload to S3
    url = "https://lab.plantnet.org/LifeCLEF/PlantCLEF2024/single_plant_training_data/PlantCLEF2024singleplanttrainingdata_800_max_side_size.tar"

    # AWS client
    # s3_client = boto3.client("s3")
    # bucket_name = "plantclef2025"
    # object_name = "PlantCLEF2024singleplanttrainingdata_800_max_side_size.tar"
    file_path = "/tmp/largefile.tar"  # Temporary local path for the file

    ###############################

    # Step 1: Download the file from the URL
    if download_from_url(url, file_path):
        print(f"File downloaded successfully to {file_path}.")
    else:
        print("Failed to download the file.")
        exit(1)

    # # Step 2: Upload the file to S3
    # if upload_local_file_to_s3(file_path, bucket_name, object_name, config):
    #     print(f"File uploaded successfully to s3://{bucket_name}/{object_name}.")
    # else:
    #     print("Failed to upload the file.")
    #     exit(1)

# Clean up the temporary file (optional)
# os.remove(file_path)
# print(f"Temporary file {file_path} removed.")
# print("File uploaded successfully!")
