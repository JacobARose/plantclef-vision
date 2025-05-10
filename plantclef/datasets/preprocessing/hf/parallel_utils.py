""" """

from datasets import Dataset, load_dataset, concatenate_datasets
import os
from tqdm.auto import tqdm
from typing import Any, Callable, Optional


def process_and_save_incrementally(
    dataset_or_path: Any,
    output_dir: str,
    processing_fn: Callable,
    batch_size: int = 100,
    save_every: int = 1000,
    num_proc: int = 32,
    keep_in_memory: bool = False,
    resume_from: Optional[int] = None,
) -> Dataset:
    """
    Process a dataset incrementally and save intermediate results.

    Args:
        dataset_or_path: Dataset object or path to load from
        output_dir: Directory to save processed batches
        processing_fn: Function to process each example
        batch_size: Number of examples to process at once
        save_every: Save after processing this many examples
        num_proc: Number of processes for parallel processing
        keep_in_memory: Whether to keep processed data in memory
        resume_from: Index to resume processing from

    Returns:
        Processed dataset
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load dataset if path is provided
    if isinstance(dataset_or_path, str):
        dataset = load_dataset(dataset_or_path)
    else:
        dataset = dataset_or_path

    # Get total size
    total_size = len(dataset)

    start_idx = resume_from or 0

    # Initialize processed datasets list
    processed_datasets = []

    # Check for existing processed batches if resuming
    if resume_from is not None:
        for i in range(0, resume_from, save_every):
            batch_path = os.path.join(
                output_dir, f"processed_batch_{i}_{i+save_every}.arrow"
            )
            if os.path.exists(batch_path):
                print(f"Loading existing batch {i}-{i+save_every}")
                batch_dataset = Dataset.load_from_disk(batch_path)
                if keep_in_memory:
                    processed_datasets.append(batch_dataset)
    else:
        resume_from = 0

    # Process in batches
    for start_idx in tqdm(range(resume_from, total_size, save_every)):
        end_idx = min(start_idx + save_every, total_size)

        # Get slice of dataset
        slice_dataset = dataset.select(range(start_idx, end_idx))

        # Process the slice
        processed_slice = slice_dataset.map(
            processing_fn,
            batched=True,
            batch_size=batch_size,
            num_proc=num_proc,
            desc=f"Processing {start_idx}-{end_idx}",
        )

        # Save the processed slice
        batch_path = os.path.join(
            output_dir, f"processed_batch_{start_idx}_{end_idx}.arrow"
        )
        processed_slice.save_to_disk(batch_path)
        print(f"Saved batch {start_idx}-{end_idx} to {batch_path}")

        # Optionally keep in memory
        if keep_in_memory:
            processed_datasets.append(processed_slice)
        else:
            # Clear memory
            del processed_slice
            import gc

            gc.collect()

    # Combine all processed datasets if kept in memory
    if keep_in_memory and processed_datasets:
        return concatenate_datasets(processed_datasets)
    else:
        # Load and combine all saved batches
        all_batches = []
        for start_idx in range(0, total_size, save_every):
            end_idx = min(start_idx + save_every, total_size)
            batch_path = os.path.join(
                output_dir, f"processed_batch_{start_idx}_{end_idx}.arrow"
            )
            if os.path.exists(batch_path):
                all_batches.append(Dataset.load_from_disk(batch_path))

        return concatenate_datasets(all_batches)
