"""
file: transforms.py
Created on Tuesday May 13th, 2025
Created by: Jacob A Rose


"""

import torch
import albumentations as A
import PIL.Image
from typing import Callable
import cv2
from torchvision.transforms import ToTensor

from plantclef.datasets.image import pil_to_numpy

cv2.setNumThreads(0)

to_tensor = ToTensor()


def get_transforms(is_training: bool = False, crop_size: int = 518) -> Callable:
    tranforms_list = [
        # A.Normalize(
        #     mean=(0.5, 0.5, 0.5),
        #     std=(0.5, 0.5, 0.5),
        #     max_pixel_value=255.0
        # )
    ]

    if is_training:
        tranforms_list.extend(
            [
                A.RandomResizedCrop(
                    size=(crop_size, crop_size),
                    scale=(0.08, 1.0),
                    ratio=(0.75, 1.33),
                    interpolation=cv2.INTER_AREA,  # cv2.INTER_LINEAR,
                ),
                A.HorizontalFlip(),
                A.VerticalFlip(),
                A.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                    max_pixel_value=255.0,
                ),
            ]
        )
    else:
        tranforms_list.extend(
            [
                A.SmallestMaxSize(max_size=crop_size, interpolation=cv2.INTER_AREA),
                A.CenterCrop(crop_size, crop_size),
                # A.Normalize(mean=0.0, std=1.0),
                A.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                    max_pixel_value=255.0,
                ),
            ]
        )
    tx = A.Compose(tranforms_list, additional_targets=None)

    def transform_func(image: PIL.Image.Image) -> torch.Tensor:
        image = pil_to_numpy(image)

        image = tx(image=image)["image"]
        return to_tensor(image)

    def collate_fn(batch):
        # print(type(batch))
        # print(len(batch))
        if isinstance(batch, list):
            return torch.stack([transform_func(image=item) for item in batch])
        else:
            # print(batch.shape)
            return transform_func(image=batch)

    return collate_fn
