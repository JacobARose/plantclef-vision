""" """

import os
from tqdm.auto import tqdm
from typing import Optional


from plantclef.datasets.preprocessing.hf.train_val_test_subsets_to_hf import (
    Config,
    preprocess_hf_dataset,
    HFDataset,
)
from plantclef.embed.utils import print_current_time, print_dir_size
from plantclef.utils.imutils import ImageProcessor

# from tqdm import tqdm
from datasets import Image
from functools import partial
from typing import Dict
import argparse


class ResizeDatasetConfig:
    def __init__(
        self,
        batch_size: int = 16,
        num_batches_per_shard: int = 4,
        image_size: Dict[str, int] = {"shortest_edge": 588},
        interpolation_mode: str = "BILINEAR",
        **kwargs,
    ):
        """Initialize the configurator with processing parameters."""
        self.batch_size = batch_size
        self.num_batches_per_shard = num_batches_per_shard
        self.image_size = image_size
        self.interpolation_mode = interpolation_mode

        # Derived configurations
        self.shard_size = batch_size * num_batches_per_shard
        self.num_shards = 0
        self.data_cfg = Config(
            image_size=image_size, interpolation_mode=interpolation_mode
        )

    def setup_processor(self) -> partial:
        """Configure and return the image processor function."""
        processor = ImageProcessor(
            image_size=self.image_size, interpolation_mode=self.interpolation_mode
        )
        return partial(processor.process_batch_with_key, key=self.data_cfg.x_col)


def process_shard(
    shard: HFDataset,
    shard_idx: int,
    total_size: int,
    cfg: ResizeDatasetConfig,
    num_proc: Optional[int] = None,
) -> None:
    """
    Process a single shard of the dataset.

    Args:
        shard: The shard to process
        shard_idx: Current shard index
        total_size: Total size of the dataset
        cfg: Configuration object
        num_proc: Number of processes to use (defaults to CPU count)
    """
    num_proc = num_proc or os.cpu_count()

    processor = ImageProcessor(cfg.image_size, cfg.interpolation_mode)
    process_func = processor.configure_processor(key=cfg.data_cfg.x_col)

    # Process the shard
    processed_shard = shard.map(
        process_func,
        input_columns=cfg.data_cfg.x_col,
        batched=True,
        batch_size=cfg.batch_size,
        num_proc=num_proc,
        desc=f"Processing shard {shard_idx} of {cfg.num_shards}",
    )

    # Transform the data
    processed_shard = processed_shard.rename_column(cfg.data_cfg.x_col, "image")
    processed_shard = processed_shard.cast_column("image", Image())

    # Save the processed shard
    shard_path = cfg.data_cfg.get_shard_path(shard_idx, cfg.shard_size, total_size)
    processed_shard.save_to_disk(shard_path)


def process_data_subset(ds: HFDataset, cfg: ResizeDatasetConfig) -> None:
    """
    Main function to process an entire dataset subset.

    Args:
        ds: The dataset subset to process
        cfg: Configuration object containing all processing parameters
    """
    # Get the dataset

    # ds = dataset_subsets["train"]

    # Calculate shards information
    total_size = len(ds)
    cfg.num_shards = total_size // cfg.shard_size + 1

    # Get starting point
    resume_from_shard = cfg.data_cfg.get_last_existing_shard_idx()
    start_shard = resume_from_shard or 0

    # Prepare and process shards
    shards = ds.batch(cfg.shard_size).skip(start_shard)

    for shard_idx, shard in tqdm(
        enumerate(shards, start=start_shard),
        total=cfg.num_shards - start_shard,
        desc=f"Processing dataset shards from {start_shard} to {cfg.num_shards}",
        unit="shard",
        unit_scale=cfg.shard_size,
    ):
        shard = HFDataset.from_dict(shard)
        process_shard(shard, shard_idx, total_size, cfg)

    print(
        f"[FINISHED] - Processing subset {cfg.data_cfg.subset} with total size = {total_size}"
    )
    print_current_time()
    print_dir_size(cfg.data_cfg.hf_dataset_path)


def process_data_subsets(cfg: ResizeDatasetConfig) -> None:
    """
    Process the dataset subsets.

    Args:
        cfg: Configuration object containing all processing parameters
    """
    dataset_subsets = preprocess_hf_dataset(cfg.data_cfg)

    for subset_name, ds in dataset_subsets.items():
        print_current_time()
        print(f"Processing {subset_name} subset with total length = {len(ds)}...")

        cfg.data_cfg.set_subset(subset_name)

        try:
            process_data_subset(ds, cfg)
        except Exception as e:
            print(f"Error processing {subset_name} subset: {e}")
            exit(1)

        print_current_time()
        print_dir_size(cfg.data_cfg.hf_dataset_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resize dataset configuration")
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size for processing"
    )
    parser.add_argument(
        "--num_batches_per_shard",
        type=int,
        default=4,
        help="Number of batches per shard",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=588,
        help="Shortest edge size for resizing images",
    )
    parser.add_argument(
        "--interpolation_mode",
        type=str,
        default="BILINEAR",
        help="Interpolation mode for resizing",
    )
    args = parser.parse_args()

    return args


def main(args: Optional[argparse.Namespace] = None) -> None:
    """
    Main function to run the processing.
    """

    args = parse_args()

    # Initialize configuration
    cfg = ResizeDatasetConfig(
        batch_size=args.batch_size,
        num_batches_per_shard=args.num_batches_per_shard,
        image_size={"shortest_edge": args.image_size},
        interpolation_mode=args.interpolation_mode,
    )

    # Process the dataset subsets
    process_data_subsets(cfg)
    print_current_time()


if __name__ == "__main__":
    main()


###################################################################################################

# def process_and_save_incrementally(
#     dataset_or_path: Any,
#     output_dir: str,
#     processing_fn: Callable,
#     batch_size: int = 100,
#     save_every: int = 1000,
#     num_proc: int = 32,
#     keep_in_memory: bool = False,
#     resume_from: Optional[int] = None,
# ) -> Dataset:
#     """
#     Process a dataset incrementally and save intermediate results.

#     Args:
#         dataset_or_path: Dataset object or path to load from
#         output_dir: Directory to save processed batches
#         processing_fn: Function to process each example
#         batch_size: Number of examples to process at once
#         save_every: Save after processing this many examples
#         num_proc: Number of processes for parallel processing
#         keep_in_memory: Whether to keep processed data in memory
#         resume_from: Index to resume processing from

#     Returns:
#         Processed dataset
#     """
#     # Create output directory if it doesn't exist
#     os.makedirs(output_dir, exist_ok=True)

#     # Load dataset if path is provided
#     if isinstance(dataset_or_path, str):
#         dataset = load_dataset(dataset_or_path)
#     else:
#         dataset = dataset_or_path

#     # Get total size
#     total_size = len(dataset)

#     start_idx = resume_from or 0

#     # Initialize processed datasets list
#     processed_datasets = []

#     # Check for existing processed batches if resuming
#     if resume_from is not None:
#         for i in range(0, resume_from, save_every):
#             batch_path = os.path.join(
#                 output_dir, f"processed_batch_{i}_{i+save_every}.arrow"
#             )
#             if os.path.exists(batch_path):
#                 print(f"Loading existing batch {i}-{i+save_every}")
#                 batch_dataset = Dataset.load_from_disk(batch_path)
#                 if keep_in_memory:
#                     processed_datasets.append(batch_dataset)
#     else:
#         resume_from = 0

#     # Process in batches
#     for start_idx in tqdm(range(resume_from, total_size, save_every)):
#         end_idx = min(start_idx + save_every, total_size)

#         # Get slice of dataset
#         slice_dataset = dataset.select(range(start_idx, end_idx))

#         # Process the slice
#         processed_slice = slice_dataset.map(
#             processing_fn,
#             batched=True,
#             batch_size=batch_size,
#             num_proc=num_proc,
#             desc=f"Processing {start_idx}-{end_idx}",
#         )

#         # Save the processed slice
#         batch_path = os.path.join(
#             output_dir, f"processed_batch_{start_idx}_{end_idx}.arrow"
#         )
#         processed_slice.save_to_disk(batch_path)
#         print(f"Saved batch {start_idx}-{end_idx} to {batch_path}")

#         # Optionally keep in memory
#         if keep_in_memory:
#             processed_datasets.append(processed_slice)
#         else:
#             # Clear memory
#             del processed_slice
#             import gc

#             gc.collect()

#     # Combine all processed datasets if kept in memory
#     if keep_in_memory and processed_datasets:
#         return concatenate_datasets(processed_datasets)
#     else:
#         # Load and combine all saved batches
#         all_batches = []
#         for start_idx in range(0, total_size, save_every):
#             end_idx = min(start_idx + save_every, total_size)
#             batch_path = os.path.join(
#                 output_dir, f"processed_batch_{start_idx}_{end_idx}.arrow"
#             )
#             if os.path.exists(batch_path):
#                 all_batches.append(Dataset.load_from_disk(batch_path))

#         return concatenate_datasets(all_batches)
