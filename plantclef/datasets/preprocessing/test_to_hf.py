"""
file: test_to_hf.py
Created on: Tuesday May 6th, 2025
Created by: Jacob A Rose


Load the directory of test images from plantclef 2025 into a Hugging Face Dataset, and write it to disk for later efficient reading during model embedding/inference.

(Added Tuesday May 6th, 2025)
    * Running this on the free lightning studio environment just took less than 60 seconds.
"""

from datasets import Image, Value, Dataset as HFDataset
from plantclef.datasets.utils import collect_image_filepaths


def create_unlabeled_hf_dataset(dataset_dir: str) -> HFDataset:
    """
    Create a Hugging Face Dataset from the images in the specified directory.

    Args:
        dataset_dir (str): The directory containing the images.

    Returns:
        HFDataset: A Hugging Face Dataset containing the images.
    """
    image_paths = collect_image_filepaths(dataset_dir)

    ds = HFDataset.from_dict({"image": image_paths, "file_path": image_paths})
    ds = ds.cast_column("image", Image())
    ds = ds.cast_column("file_path", Value("string"))

    return ds


def main():
    dataset_dir = "/teamspace/studios/this_studio/plantclef-vision/data/plantclef2025/competition-metadata/PlantCLEF2025_test_images/PlantCLEF2025_test_images"
    output_dir = "/teamspace/studios/this_studio/plantclef-vision/data/parquet/plantclef2025/full_test/HF_dataset"

    ds = create_unlabeled_hf_dataset(dataset_dir)

    print(f"Saving dataset found at {dataset_dir}\nto\n{output_dir}")
    print(f"Dataset size: {len(ds)}")
    print(f"Dataset columns: {ds.column_names}")
    print(f"Dataset features: {ds.features}")
    ds.save_to_disk(output_dir)


if __name__ == "__main__":
    main()
