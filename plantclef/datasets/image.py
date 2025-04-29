"""
image.py

This module contains functions and classes for processing image datasets.
It includes utilities for loading, transforming, and augmenting image data
to prepare it for machine learning workflows.

Author: Jacob A Rose
Date: Saturday Apr 19th, 2025
"""

from multiprocessing import Pool
from typing import List, Tuple
from PIL import Image
from PIL.Image import Resampling
import torch
from torchvision.transforms.functional import pil_to_tensor
from tqdm import tqdm
from functools import partial
from torchvision import transforms as T

resampling_modes = {
    "nearest": Resampling.NEAREST,
    "box": Resampling.BOX,
    "bilinear": Resampling.BILINEAR,
    "hamming": Resampling.HAMMING,
    "bicubic": Resampling.BICUBIC,
    "lanczos": Resampling.LANCZOS,
}


def smart_resize(
    image: Image.Image, target_height: int, target_width: int, mode: str = "nearest"
) -> Image.Image:
    """
    Resizes an image to the target dimensions while maintaining aspect ratio.
    Pads the image with black (zero) pixels to fit the target dimensions if necessary.

    :param image: Input PIL.Image object
    :param target_height: Target height of the resized image
    :param target_width: Target width of the resized image
    :return: Resized and padded PIL.Image object
    """
    # Get original dimensions
    original_width, original_height = image.size

    # Calculate the scaling factor to maintain aspect ratio
    scale = min(target_width / original_width, target_height / original_height)
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)

    # Resize the image while maintaining aspect ratio
    resized_image = image.resize((new_width, new_height), resampling_modes[mode])

    # Create a new black (zero) image with the target dimensions
    new_image = Image.new("RGB", (target_width, target_height), (0, 0, 0))

    # Calculate the position to center the resized image
    top_left_x = (target_width - new_width) // 2
    top_left_y = (target_height - new_height) // 2

    # Paste the resized image onto the black canvas
    new_image.paste(resized_image, (top_left_x, top_left_y))

    return new_image

pil_to_tensor = T.ToTensor()

def parse_image(
    file_path: str, target_height: int, target_width: int, mode: str = "nearest"
) -> Tuple[str, torch.Tensor]:
    """
    Resizes an image using smart_resize and returns the resized image in memory.

    :param file_path: Path to the input image file
    :param target_height: Target height for resizing
    :param target_width: Target width for resizing
    :return: A tuple containing the file path and the resized PIL.Image object
    """
    try:
        # Open the image
        image = Image.open(file_path)

        # Apply smart_resize
        resized_image = smart_resize(image, target_height, target_width, mode)

        return file_path, pil_to_tensor(resized_image)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return file_path, None


def resize_images_in_parallel(
    file_paths: List[str],
    target_height: int,
    target_width: int,
    mode: str = "nearest",
    num_workers: int = 4,
) -> List[Tuple[str, Image.Image]]:
    """
    Maps the smart_resize function to a list of image file paths in parallel using multiprocessing.Pool.imap.

    :param file_paths: List of image file paths
    :param target_height: Target height for resizing
    :param target_width: Target width for resizing
    :param num_workers: Number of worker processes to use for parallel processing
    :return: A list of tuples containing file paths and resized PIL.Image objects
    """
    resized_images = []
    with Pool(processes=num_workers) as pool:
        process_with_params = partial(
            parse_image,
            target_height=target_height,
            target_width=target_width,
            mode=mode,
        )

        # Use tqdm to display a progress bar
        for result in tqdm(
            pool.imap(process_with_params, file_paths),
            total=len(file_paths),
            desc="Resizing images",
            unit="image",
        ):
            resized_images.append(result)

    paths, imgs = zip(*resized_images)
    return paths, imgs


############## Image Dimension Calculation


def parse_image_dimensions(file_path: str) -> Tuple[str, int, int]:
    """
    Reads an image file and returns its file path, height, and width.

    :param file_path: Path to the image file
    :return: A tuple containing the file path, height, and width of the image
    """
    try:
        with Image.open(file_path) as img:
            width, height = img.size
        return file_path, width, height
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return file_path, None, None


def calculate_image_dimensions(
    file_paths: List[str], num_workers: int = 4
) -> List[Tuple[str, int, int]]:
    """
    Calculates the dimensions (height and width) of images in parallel using multiprocessing.

    :param file_paths: List of image file paths
    :param num_workers: Number of worker processes to use for parallel processing
    :return: A list of tuples containing file paths, heights, and widths of the images
    """
    dimensions = []
    with Pool(processes=num_workers) as pool:
        # Use tqdm to display a progress bar
        for result in tqdm(
            pool.imap(parse_image_dimensions, file_paths),
            total=len(file_paths),
            desc="Calculating image dimensions",
            unit="image",
        ):
            dimensions.append(result)
    return dimensions
