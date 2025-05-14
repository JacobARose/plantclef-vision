"""
file: transforms.py
Created on Tuesday May 13th, 2025
Created by: Jacob A Rose


"""

import torch
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import numpy as np
import PIL.Image
from typing import List, Callable
import cv2


def get_transforms(is_training: bool = False) -> Callable:
    tx = [
        # A.Normalize(
        #     mean=(0.5, 0.5, 0.5),
        #     std=(0.5, 0.5, 0.5),
        #     max_pixel_value=255.0
        # )
    ]

    if is_training:
        tx.extend(
            *[
                A.RandomResizedCrop(
                    size=(518, 518),
                    scale=(0.08, 1.0),
                    ratio=(0.75, 1.33),
                    interpolation=cv2.INTER_LINEAR,
                ),
                A.HorizontalFlip(),
                A.VerticalFlip(),
                # A.Normalize(
                #     mean=(0.485, 0.456, 0.406),
                #     std=(0.229, 0.224, 0.225)
                # )
            ]
        )
    else:
        tx.extend(
            [
                A.SmallestMaxSize(max_size=518, interpolation=cv2.INTER_AREA),
                A.CenterCrop(518, 518),
                # A.Normalize(mean=0.0, std=1.0),
                A.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                    max_pixel_value=255.0,
                ),
                ToTensorV2(),
            ]
        )
    tx = A.Compose(tx, additional_targets=None)

    def transform_func(image: List[PIL.Image.Image]) -> np.ndarray:
        image = np.array(image)
        print(image.dtype)
        # image = image.squeeze()

        image = tx(image=image)["image"]
        return image

    def collate_fn(batch):
        return torch.stack([transform_func(item) for item in batch])

    return collate_fn
