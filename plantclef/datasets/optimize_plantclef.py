"""
Created on Saturday Apr 27th, 2025
Create by: Jacob A Rose


This module contains functions used to prepare data for passing into the
`litdata.optimize` module for serialization. The functions in this file
are designed to transform, validate, and structure data to ensure it is
ready for optimization and subsequent serialization processes.
"""

# from litdata import clear_cache
from litdata import optimize
from functools import partial
from pprint import pprint
from plantclef.datasets.image import parse_image
from PIL.Image import Image
from lightning import seed_everything
from typing import Any, Dict, Optional, Tuple
from pathlib import Path
import pandas as pd
from plantclef.datasets.utils import (
    merge_filepaths_with_metadata,
    clear_cache,
    load_plantclef_class2idx,
    optimize_pandas_dtypes,
)
import os


def load_metadata(
    metadata_path: Path,
    dataset_dir: Path,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Load the metadata csv file and merge with the image file paths.
    """

    metadata_dir = metadata_path.parent
    metadata_name = metadata_path.stem

    metadata_cache_path = Path(metadata_dir, metadata_name + ".parquet")

    metadata = None
    if metadata_cache_path.exists():
        print(
            "Found previously preprocessed metadata cache, loading from parquet file and skipping preprocessing"
        )
        try:
            metadata = pd.read_parquet(metadata_cache_path)
        except Exception as e:
            print(
                "Failed to load parquet file. Attempting to load from csv instead. Error message:\n",
                e,
            )
    if metadata is None:
        print(
            "No valid previously preprocessed metadata cache found, loading from CSV and performing preprocessing."
        )
        metadata = pd.read_csv(metadata_path, sep=";", low_memory=False)

        metadata = merge_filepaths_with_metadata(
            metadata_df=metadata, dataset_dir=dataset_dir
        )

        metadata = optimize_pandas_dtypes(metadata)

        print("COMPLETE - Preprocessing metadata csv.")
        print(
            f"Saving metadata csv to a parquet cache for faster retrieval. File is located at:\n{metadata_cache_path}"
        )
        metadata.to_parquet(metadata_cache_path)
        print(
            f"COMPLETE - Saved metadata csv to a parquet cache for faster retrieval. File is located at:\n{metadata_cache_path}"
        )

    plantclef_class_index_path = Path(metadata_dir, "species_ids.csv")
    class2idx = load_plantclef_class2idx(plantclef_class_index_path)

    return metadata, class2idx


def get_inputs(
    data_df: pd.DataFrame,
    path_col: str = "image_path",
    label_col: str = "species_id",
    label_encoder: Optional[Dict] = None,
) -> Any:
    """ """

    if label_encoder is None:
        df = data_df.assign(label_idx=data_df[label_col])
    else:
        df = data_df.assign(
            label_idx=data_df[label_col].map(lambda x: label_encoder[x])
        )

    rows = [
        (row[path_col], row["label_idx"], row[label_col], sample_idx)
        for sample_idx, row in df.iterrows()
    ]

    return rows


def optimize_row_fn(
    row: Tuple[str, int, int, int],
    target_height: int = 512,
    target_width: int = 512,
    mode: str = "nearest",
    **kwargs,
) -> Tuple[Image, int, int]:
    """
    Function for parsing individual rows during the litdata.optimize() process.
    Args:
        row: Tuple containing the file path, class label index, class label, and sample index.
        target_height: Target height for resizing the image.
        target_width: Target width for resizing the image.
        mode: Resizing mode (default is "nearest").


    """

    file_path, class_label_idx, class_label, sample_idx = row

    _, img = parse_image(file_path, target_height, target_width, mode)

    return img, class_label_idx, sample_idx


if __name__ == "__main__":
    print(f"Running __main__ section of python script: {__file__}")

    dataset_dir = Path(
        "/teamspace/studios/plantclef2025/plantclef-vision/data/plantclef2025/PlantCLEF2024singleplanttrainingdata_800_max_side_size/images_max_side_800"
    )

    # File path of the competition-provided csv train metadata
    train_metadata_path = Path(
        "/teamspace/studios/plantclef2025/plantclef-vision/data/plantclef2025/competition-metadata/PlantCLEF2024_single_plant_training_metadata.csv"
    )

    args = {
        "target_height": 618,
        "target_width": 618,
        "mode": "nearest",
        "output_dir": "/teamspace/studios/this_studio/plantclef-vision/data/plantclef2025/optimized/res618",
    }
    print("Arguments:")
    pprint(args)

    os.makedirs(args["output_dir"], exist_ok=True)
    seed_everything(42)
    cache_dir = "/cache/data"
    clear_cache(cache_dir)

    data_df, class2idx = load_metadata(
        metadata_path=train_metadata_path, dataset_dir=dataset_dir
    )

    inputs = get_inputs(
        data_df, path_col="image_path", label_col="species_id", label_encoder=class2idx
    )

    optimize(
        fn=partial(optimize_row_fn, args=args),
        inputs=inputs,
        output_dir=args["output_dir"],
        # chunk_size=2000,
        chunk_bytes="64MB",
        reorder_files=False,
        num_workers=os.cpu_count(),
        # num_downloaders=10,
        fast_dev_run=False,  # True,
        mode="overwrite",
    )

    clear_cache(cache_dir)
    print("Done! Check for optimized results in --> ", args["output_dir"])
