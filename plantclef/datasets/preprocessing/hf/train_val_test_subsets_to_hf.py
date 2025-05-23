"""
file: train_val_test_subsets_to_hf.py
Created on: Thursday May 8th, 2025
Created by: Jacob A Rose


Load train, val, and test subsets of the single-label full train set from disk, convert to Hugging Face Dataset, then save_to_disk for later efficient reading during model embedding/inference/training.

* num_batches_per_shard = os.cpu_count() by default

Example:

python "/teamspace/studios/this_studio/plantclef-vision/plantclef/datasets/preprocessing/hf/train_val_test_subsets_to_hf.py"


python "/teamspace/studios/this_studio/plantclef-vision/plantclef/datasets/preprocessing/hf/train_val_test_subsets_to_hf.py" --batch_size 500

python "/teamspace/studios/this_studio/plantclef-vision/plantclef/datasets/preprocessing/hf/train_val_test_subsets_to_hf.py" --batch_size 256

python "/teamspace/studios/this_studio/plantclef-vision/plantclef/datasets/preprocessing/hf/train_val_test_subsets_to_hf.py" --batch_size 128 --num_batches_per_shard 64 --image_size 350


python "/teamspace/studios/this_studio/plantclef-vision/plantclef/datasets/preprocessing/hf/train_val_test_subsets_to_hf.py" --batch_size 256 --num_batches_per_shard 32 --image_size 350

python "/teamspace/studios/this_studio/plantclef-vision/plantclef/datasets/preprocessing/hf/train_val_test_subsets_to_hf.py" --batch_size 64 --num_batches_per_shard 512 --image_size 350


###################

# python "/teamspace/studios/this_studio/plantclef-vision/plantclef/datasets/preprocessing/hf/train_val_test_subsets_to_hf.py" --batch_size 32 --num_batches_per_shard 1024 --image_size 350

python "/teamspace/studios/this_studio/plantclef-vision/plantclef/datasets/preprocessing/hf/train_val_test_subsets_to_hf.py" --batch_size 64 --num_batches_per_shard 1024 --image_size 350


python "/teamspace/studios/this_studio/plantclef-vision/plantclef/datasets/preprocessing/hf/train_val_test_subsets_to_hf.py" --batch_size 128 --num_batches_per_shard 1024 --image_size 350

python "/teamspace/studios/this_studio/plantclef-vision/plantclef/datasets/preprocessing/hf/train_val_test_subsets_to_hf.py" --batch_size 128 --num_batches_per_shard 512 --image_size 350

python "/teamspace/studios/this_studio/plantclef-vision/plantclef/datasets/preprocessing/hf/train_val_test_subsets_to_hf.py" --batch_size 128 --num_batches_per_shard 256 --image_size 322


# -- 4 core

python "/teamspace/studios/this_studio/plantclef-vision/plantclef/datasets/preprocessing/hf/train_val_test_subsets_to_hf.py" --batch_size 64 --num_batches_per_shard 128 --image_size 322



python "/teamspace/studios/this_studio/plantclef-vision/plantclef/datasets/preprocessing/hf/train_val_test_subsets_to_hf.py" --batch_size 64 --num_batches_per_shard 64 --image_size 256

# ----
# new argparser based on num_samples_per_shard instead of num_batches_per_shard:


python "/teamspace/studios/this_studio/plantclef-vision/plantclef/datasets/preprocessing/hf/train_val_test_subsets_to_hf.py" --batch_size 32 --num_samples_per_shard 4096 --image_size 256 --resume

# ---- 32-core machine

python "/teamspace/studios/this_studio/plantclef-vision/plantclef/datasets/preprocessing/hf/train_val_test_subsets_to_hf.py" --batch_size 64 --num_samples_per_shard 32768 --image_size 308


python "/teamspace/studios/this_studio/plantclef-vision/plantclef/datasets/preprocessing/hf/train_val_test_subsets_to_hf.py" --batch_size 96 --num_samples_per_shard 32768 --image_size 308 --resume



"""

import os

import pandas as pd
from pathlib import Path
from plantclef.datasets.utils import (
    merge_filepaths_with_metadata,
    load_plantclef_idx2class,
    load_plantclef_class2idx,
    optimize_pandas_dtypes,
)
from plantclef.embed.utils import print_current_time
from typing import Dict, Optional, Callable, List, Union, Any
from concurrent.futures import ThreadPoolExecutor
import torch
import PIL.Image

from plantclef.config import BaseConfig
from dataclasses import dataclass, field
from rich.repr import auto
from torchvision import transforms
from datasets import Dataset as HFDataset, DatasetDict as HFDatasetDict, Image


def default_image_size():
    return {"shortest_edge": 588}


@dataclass
@auto
class Config(BaseConfig):
    name: str = "train_val_test"
    x_col: str = "image_path"
    label_col: str = "species_id"
    target_col: str = "label_idx"

    image_size: Dict[str, int] = field(
        default_factory=default_image_size
    )  # Assuming model was trained on res=518 images with patch size 14, this keeps 5*14 = 70 extra pixels for downstream data augmentation.
    interpolation_mode: str = "nearest"

    dataset_dir: str = "/teamspace/studios/this_studio/plantclef-vision/data/plantclef2025/PlantCLEF2024singleplanttrainingdata_800_max_side_size/images_max_side_800"
    metadata_path = "/teamspace/studios/this_studio/plantclef-vision/data/plantclef2025/competition-metadata/PlantCLEF2024_single_plant_training_metadata.csv"
    metadata_cache_path: str = ""
    hf_datasets_root_dir: str = (
        "/teamspace/studios/this_studio/plantclef-vision/data/hf"
    )
    hf_dataset_dir: str = ""
    _hf_dataset_path: str = ""
    subset: Optional[str] = None

    def __post_init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print(self.image_size)

        self.hf_dataset_dir = (
            f"{self.hf_datasets_root_dir}/plantclef2025/single_label_train_val_test"
        )
        # self.hf_dataset_path = (
        #     f"{self.hf_dataset_dir}/shortest_edge_{self.image_size['shortest_edge']}"
        # )
        self.metadata_cache_path = (
            f"{Path(self.metadata_path).parent}/{Path(self.metadata_path).stem}.parquet"
        )

        self.class_index_path = (
            f"{Path(self.metadata_path).parent}/{self.label_col}s.csv"
        )
        self.set_subset(subset=self.subset)

    @property
    def hf_dataset_path(self):
        return self._hf_dataset_path

    def set_subset(self, subset: Optional[str] = None):
        """
        Set the subset for the dataset. Dynamically updates return value for self.hf_dataset_path.
        Args:
            subset (str): The subset to set. Can be 'train', 'val', or 'test'.
        """
        self.subset = subset

        if self.subset is None:
            self._hf_dataset_path = f"{self.hf_dataset_dir}/shortest_edge_{self.image_size['shortest_edge']}"
        else:
            self._hf_dataset_path = f"{self.hf_dataset_dir}/shortest_edge_{self.image_size['shortest_edge']}/{self.subset}"

    def metadata_cache_exists(self) -> bool:
        """
        Check if the metadata cache exists.
        """
        return os.path.exists(self.metadata_cache_path)

    def load_metadata(self) -> pd.DataFrame:
        """
        Load the metadata csv file and merge with the image file paths.
        """
        if self.metadata_cache_exists():
            print(
                "[Cache Found] previously preprocessed metadata cache, loading from cache file and skipping preprocessing"
            )
            try:
                metadata = pd.read_parquet(self.metadata_cache_path)
            except Exception as e:
                print(
                    "Failed to load parquet file. Attempting to load from csv instead. Error message:\n",
                    e,
                )
        else:
            print(
                "[Cache NOT Found] -- No valid previously preprocessed metadata cache found, loading from CSV + performing preprocessing."
            )
            metadata = pd.read_csv(self.metadata_path, sep=";", low_memory=False)
            metadata = merge_filepaths_with_metadata(
                metadata_df=metadata, dataset_dir=self.dataset_dir
            )
            metadata = optimize_pandas_dtypes(metadata)

            print("[PREPROCESSING COMPLETE] - Preprocessed metadata csv.")
            print(
                f"[INFO] -- Saving metadata csv to a parquet cache for faster retrieval. File is located at:\n{self.metadata_cache_path}"
            )
            metadata.to_parquet(self.metadata_cache_path)
            print(
                f"[SAVING CACHE COMPLETE] - Saved metadata csv to a parquet cache for faster retrieval. File is located at:\n{self.metadata_cache_path}"
            )

        return metadata

    def load_class_index(self, mode: str = "class2idx") -> Dict[int, int]:
        if mode == "class2idx":
            label_encoder = load_plantclef_class2idx(self.class_index_path)
        elif mode == "idx2class":
            label_encoder = load_plantclef_idx2class(self.class_index_path)
        else:
            raise Exception(
                f"Argument Error: arg mode={mode} must be either 'class2idx' or 'idx2class'"
            )

        return label_encoder

    def encode_target_col(
        self,
        data_df: pd.DataFrame,
        class2idx: Optional[Dict[int, int]] = None,
        label_col: Optional[str] = None,
        target_col: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Creates the `self.target_col` column in the metadata data_df.

        If class2idx mapping dict is provided, fills each row of `self.target_col` with the value of class2idx corresponding to the key in that row's `self.label_col`.

        If class2idx is not provided, data_df[self.target_col] will be exactly equal to data_df[self.label_col].

        """
        label_col = label_col or self.label_col
        target_col = target_col or self.target_col

        if class2idx is None:
            data_df = data_df.assign(label_idx=data_df[label_col])
        else:
            data_df = data_df.assign(
                label_idx=data_df[label_col].map(lambda x: class2idx[x])
            )
        return data_df

    def get_shard_path(self, shard_idx: int, shard_size: int, total_size: int) -> str:
        """
        Helper function for getting a formatted arrow file path str given the
            shard_idx: int - Current shard index number
            shard_size: int - The number of samples per shard
            total_size: int - The total number of samples across all shards
        """

        return os.path.join(
            self.hf_dataset_path,
            f"data_shard_{shard_idx}_{shard_idx * shard_size}_{min((shard_idx + 1) * shard_size, total_size)}.arrow",
        )

    def get_shard_idx(self, shard_path: str) -> int:
        """
        Helper function for getting the shard index from a formatted arrow file path str
        """

        # Extract the shard index from the filename
        shard_idx = int(Path(shard_path).stem.split("_")[2])

        return shard_idx

    def get_last_existing_shard_idx(self) -> int:
        """
        Helper function for getting the last existing shard index from the dataset directory to continue from a checkpoint
        """
        last_existing_shard_idx = 0
        if not os.path.exists(self.hf_dataset_path):
            return 0

        found_shards = self.get_shard_paths()
        found_shard_idxs = [self.get_shard_idx(p) for p in found_shards]
        if len(found_shard_idxs) > 0:
            last_existing_shard_idx = max(found_shard_idxs) + 1

        return last_existing_shard_idx

    def get_shard_paths(self) -> List[str]:
        """
        Helper function for getting the shard paths from the dataset directory
        """
        if not os.path.exists(self.hf_dataset_path):
            return []

        found_shards = sorted(
            [p for p in os.listdir(self.hf_dataset_path) if p.endswith(".arrow")]
        )
        found_shard_paths = [
            os.path.join(self.hf_dataset_path, p) for p in found_shards
        ]

        return found_shard_paths


def get_transforms(
    image_size: Optional[Dict[str, int]] = None,
    crop_size: Optional[Dict[str, int]] = None,
) -> Callable:
    image_size = image_size or {"shortest_edge": 518}
    crop_size = crop_size or {"height": 518, "width": 518}
    tx = []
    tx.append(
        transforms.Resize(
            size=image_size["shortest_edge"],
            interpolation=transforms.InterpolationMode.BICUBIC,
            max_size=None,
            antialias=True,
        )
    )

    if isinstance(crop_size, dict):
        if "height" in crop_size and "width" in crop_size:
            if crop_size["height"] <= image_size["shortest_edge"]:
                tx.append(
                    transforms.CenterCrop(
                        size=(crop_size["height"], crop_size["width"])
                    )
                )

    return transforms.Compose(tx)


def get_dict_transform(transform_kwargs={}, input_columns=None) -> Callable:
    tx = get_transforms(**transform_kwargs)

    def func(data, *args, **kwargs):
        if (input_columns is not None) and isinstance(input_columns, str):
            if isinstance(data, dict):
                data = data[input_columns]
            return {input_columns: tx(data)}
        return tx(data)

    return func


def transform_image_batch(
    batch: List[str],
    transform: Callable,
    return_as_dict_key: Optional[str] = None,
    num_threads: Optional[int] = None,
) -> Union[List[Any], Dict[str, List[Any]]]:
    """Process a batch of images using multiple threads."""
    results = []

    def process_single_image(path: str) -> torch.Tensor:
        try:
            with PIL.Image.open(path) as img:
                # Convert to RGB if not already
                if img.mode != "RGB":
                    img = img.convert("RGB")
                # Apply transform and convert to tensor
                tensor = transform(img)
                return tensor
        except Exception as e:
            print(f"Error processing {path}: {e}")
            return torch.zeros((3, 588, 588))  # Return dummy tensor on error

    num_threads = num_threads or 0

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = {executor.submit(process_single_image, path): path for path in batch}
        for future in futures:
            results.append(future.result())
    if isinstance(return_as_dict_key, str):
        results = {return_as_dict_key: results}
    return results


def preprocess_hf_dataset(cfg):
    class2idx = cfg.load_class_index(mode="class2idx")
    metadata = cfg.load_metadata()
    metadata = cfg.encode_target_col(metadata, class2idx=class2idx)
    keep_cols = [
        "image_path",
        "label_idx",
        "image_name",
        "organ",
        "species_id",
        "obs_id",
        "author",
        "altitude",
        "latitude",
        "longitude",
        "species",
        "genus",
        "family",
        "learn_tag",
    ]

    metadata = metadata[keep_cols]

    train_df = metadata[metadata["learn_tag"] == "train"]
    val_df = metadata[metadata["learn_tag"] == "val"]
    test_df = metadata[metadata["learn_tag"] == "test"]

    train_ds = HFDataset.from_pandas(train_df)
    val_ds = HFDataset.from_pandas(val_df)
    test_ds = HFDataset.from_pandas(test_df)

    dataset = HFDatasetDict({"train": train_ds, "val": val_ds, "test": test_ds})

    return dataset


def create_single_label_hf_dataset(
    cfg: Optional[Config] = None, batch_size: Optional[int] = None
) -> HFDatasetDict:
    """ """

    cfg = cfg or Config()

    dataset = preprocess_hf_dataset(cfg)

    tx = get_dict_transform(
        transform_kwargs={"image_size": {"shortest_edge": 716}}, input_columns=cfg.x_col
    )

    if batch_size is None:
        print(f"[INITIATING dataset.map(resize)] -- using num_proc={os.cpu_count()}")
        dataset = dataset.cast_column(cfg.x_col, Image())
        dataset = dataset.map(tx, input_columns=cfg.x_col, num_proc=os.cpu_count())
    else:
        from functools import partial

        num_threads = os.cpu_count() or 0
        num_threads = num_threads // 2
        tx = partial(
            transform_image_batch,
            transform=tx,
            return_as_dict_key=cfg.x_col,
            num_threads=num_threads,
        )
        print(
            f"[INITIATING dataset.map(resize)] -- using num_proc={os.cpu_count()} and num_threads={num_threads},",
            f"batch_size={batch_size}",
        )
        dataset = dataset.map(
            tx,
            input_columns=cfg.x_col,
            num_proc=os.cpu_count(),
            batched=True,
            batch_size=batch_size,
        )
    print("[ds.map(Resize) COMPLETE]")
    print_current_time()

    return dataset


# def main(cfg: Optional[Config] = None) -> None:
#     cfg = cfg or Config()
#     cfg.show()

#     # Creating the resized image HFDataset with metadata columns
#     dataset = create_single_label_hf_dataset(cfg, batch_size=500)

#     print("[INITIATING dataset.save_to_disk()]")

#     dataset.save_to_disk(cfg.hf_dataset_path, num_proc=os.cpu_count())

#     print("[Dataset.save_to_disk() COMPLETE]")
#     print_current_time()
#     print_dir_size(cfg.hf_dataset_path)


if __name__ == "__main__":
    from plantclef.datasets.preprocessing.hf.parallel_utils import main

    main()
