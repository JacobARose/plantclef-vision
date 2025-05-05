"""
File: test_dataloader.py
Author: Jacob Alexander Rose
Created on Saturday May 3rd, 2025


"""

import argparse
from typing import Callable, Optional
import time
from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader
import torch
from torchvision import transforms
import numpy as np
from tqdm import tqdm


def benchmark_dataloader(
    dataloader, num_batches: int = 100, warmup: int = 5, position: int = 0
):
    """
    Benchmark the dataloader for a specified number of batches.

    Args:
        dataloader: PyTorch DataLoader instance
        num_batches: Number of batches to measure
        warmup: Number of warmup iterations to ignore
        position: Position for tqdm progress bar

    Returns:
        Dictionary containing timing statistics
    """
    # Warm up
    for _ in range(warmup):
        next(iter(dataloader))

    times = []
    start_time = time.perf_counter()

    for i, batch in tqdm(
        enumerate(dataloader),
        total=num_batches,
        desc="Benchmarking DataLoader",
        position=position,
    ):
        if i >= num_batches:
            break

        batch_start = time.perf_counter()

        # Force data transfer
        if isinstance(batch, dict):
            for v in batch.values():
                if isinstance(v, torch.Tensor):
                    v.cpu()
        elif isinstance(batch, (tuple, list)):
            for item in batch:
                if isinstance(item, torch.Tensor):
                    item.cpu()

        batch_end = time.perf_counter()
        times.append(batch_end - batch_start)

    end_time = time.perf_counter()

    return {
        "total_time": end_time - start_time,
        "avg_batch_time": np.mean(times),
        "std_batch_time": np.std(times),
        "median_batch_time": np.median(times),
        "min_batch_time": np.min(times),
        "max_batch_time": np.max(times),
    }


def compare_configurations(
    dataset_path="glue",
    dataset_name="sst2",
    split="train",
    batch_sizes=[32, 64],
    num_workers_list=[0, 2, 4],
    num_batches: int = 100,
    warmup: int = 5,
    image_size: int = 512,
):
    """
    Compare different DataLoader configurations.

    Args:
        dataset_path: Name of the HuggingFace dataset
        dataset_name: Configuration of the dataset
        split: Dataset split to use
        batch_sizes: List of batch sizes to test
        num_workers_list: List of num_workers values to test
    """
    # Load dataset
    try:
        dataset = load_dataset(dataset_path, dataset_name, split=split)
    except Exception:
        dataset = Dataset.load_from_disk(dataset_path)
    dataset = dataset.with_format("torch")

    transform = create_transform(image_size, key="image")
    dataset.set_transform(transform)

    results = {}

    print(f"Running Benchmark on {dataset_path} - {dataset_name} ({split})")
    print("Exploring configuration parameters:")
    print(f"Batch Sizes: {batch_sizes}")
    print(f"Num Workers: {num_workers_list}")
    print(f"Number of batches: {num_batches}")
    print(f"Warmup: {warmup}")

    for batch_size in tqdm(batch_sizes, desc="Batch Sizes", position=0):
        for num_workers in tqdm(num_workers_list, desc="Num Workers", position=1):
            config_name = f"bs_{batch_size}_nw_{num_workers}"

            # Create DataLoader
            dl = DataLoader(
                dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True
            )

            # Run benchmark
            stats = benchmark_dataloader(
                dl, num_batches=num_batches, warmup=warmup, position=2
            )

            results[config_name] = {
                "configuration": {"batch_size": batch_size, "num_workers": num_workers},
                "performance": stats,
            }

    return results


def print_results(results):
    """Print benchmark results in a readable format."""
    for config_name, result in results.items():
        print(f"\nConfiguration: {result['configuration']}")
        perf = result["performance"]
        print(f"Total time: {perf['total_time']:.3f}s")
        print(f"Average batch time: {perf['avg_batch_time']*1000:.2f}ms")
        print(f"Median batch time: {perf['median_batch_time']*1000:.2f}ms")
        print(
            f"Min/MAX batch time: {perf['min_batch_time']*1000:.2f}ms/{perf['max_batch_time']*1000:.2f}ms"
        )


################################
#################################


def create_transform(image_size: int, key: Optional[str] = None) -> Callable:
    """Create image transformation pipeline that maintains aspect ratio."""
    transform_list = [
        # transforms.ToPILImage(),
        transforms.Resize(
            image_size, max_size=image_size + 2
        ),  # Maintains aspect ratio
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
    ]
    transform_list = transforms.Compose(transform_list)
    if key is not None:
        return transform_dict(transform_list, key)
    return transform_list


def transform_dict(transforms: Callable, key: str) -> Callable:
    """Apply transformation to a specific key in the dataset."""

    def transform_fn(row):
        row[key] = [transforms(image) for image in row[key]]
        return row

    return transform_fn


def parse_arguments():
    """Parse command line arguments for benchmark configuration."""
    parser = argparse.ArgumentParser(
        description="Benchmark HuggingFace dataset loading with PyTorch DataLoader"
    )

    parser.add_argument(
        "--dataset-path",
        type=str,
        default="/teamspace/studios/this_studio/plantclef-vision/data/parquet/plantclef2025/full_test/HF_dataset",
        help="Name of the HuggingFace dataset to benchmark",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default=None,
        help="Configuration name for the dataset",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        choices=["train", "validation", "test"],
        help="Dataset split to use for benchmarking",
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[16, 32, 64],
        help="List of batch sizes to test (space-separated numbers)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        nargs="+",
        default=[0, 2, 4],
        help="List of num_workers values to test (space-separated numbers)",
    )
    parser.add_argument(
        "--num-batches", type=int, default=100, help="Number of batches to benchmark"
    )
    parser.add_argument(
        "--warmup", type=int, default=5, help="Number of warmup iterations to ignore"
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=512,
        help="Target image size for transformations",
    )

    return parser.parse_args()


def main():
    """Main function to run the benchmark with command line arguments."""
    args = parse_arguments()

    # Convert argparse lists to Python lists
    batch_sizes = list(args.batch_sizes)
    num_workers_list = list(args.num_workers)

    # Run the benchmark
    results = compare_configurations(
        dataset_path=args.dataset_path,
        dataset_name=args.dataset_name,
        split=args.split,
        batch_sizes=batch_sizes,
        num_workers_list=num_workers_list,
        num_batches=args.num_batches,
        warmup=args.warmup,
        image_size=args.image_size,
    )

    print_results(results)


if __name__ == "__main__":
    main()
