"""
Created on Sunday Apr 27th, 2025
Created by: Jacob A Rose

Based on https://github.com/Lightning-AI/litData/blob/main/examples/getting_started/stream.py

"""

from pathlib import Path
from litdata import StreamingDataLoader, StreamingDataset
from typing import Union


def test_streaming_dataset(input_dir: Union[Path, str]):
    """
    Test the Streaming Dataset and DataLoader.
    Args:
        input_dir: The directory where the optimized dataset is stored.
    """

    print(f"Testing Streaming Dataset and DataLoader with input directory: {input_dir}")

    # Create the Streaming Dataset
    dataset = StreamingDataset(str(input_dir), shuffle=True)

    # Access any elements of the dataset
    # sample = dataset[50]
    # img = sample["image"]
    # cls = sample["class"]

    # Create dataLoader and iterate over it to train your AI models.
    dataloader = StreamingDataLoader(dataset)
    for i, batch in enumerate(dataloader):
        print(f"Batch: {i}")
        print(
            f"index: {batch['index']}",
            f"image tensor.shape: {batch['image'].shape}",
            f"image tensor.dtype: {batch['image'].dtype}",
            f"image tensor.device: {batch['image'].device}",
            f"class: {batch['class']}",
            sep="\n",
        )
        if i == 5:
            break


if __name__ == "__main__":
    # Remote path where full dataset is stored

    cache_dir = "/cache"
    input_dir = str(Path(cache_dir, "my_optimized_dataset"))

    test_streaming_dataset(input_dir)
