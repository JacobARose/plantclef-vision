"""
file: imutils.py
Created on: Saturday May 10th, 2025
Created by: Jacob A Rose



"""

from typing import Dict, Any, Callable
import PIL.Image
from torchvision import transforms
from functools import partial


class ImageProcessor:
    interpolation_modes: Dict[str, Any] = {
        "BICUBIC": transforms.InterpolationMode.BICUBIC,
        "BILINEAR": transforms.InterpolationMode.BILINEAR,
        "BOX": transforms.InterpolationMode.BOX,
        "HAMMING": transforms.InterpolationMode.HAMMING,
        "LANCZOS": transforms.InterpolationMode.LANCZOS,
        "NEAREST": transforms.InterpolationMode.NEAREST,
        "NEAREST_EXACT": transforms.InterpolationMode.NEAREST_EXACT,
    }

    def __init__(self, image_size: Dict[str, int], interpolation_mode: str = "BICUBIC"):
        super().__init__()
        self.image_size = image_size
        self.interpolation_mode = self.interpolation_modes[interpolation_mode.upper()]
        self.setup()

    def setup(self):
        self.resize_tx = transforms.Resize(
            size=self.image_size["shortest_edge"],
            interpolation=self.interpolation_mode,
            max_size=None,
            antialias=True,
        )

    @classmethod
    def read_image(cls, path):
        # with PIL.Image.open(path) as img:
        #     return img
        img = PIL.Image.open(path)
        return img

    def resize_image(self, image):
        return self.resize_tx(image)

    def process_func(self, path):
        """ """

        try:
            img = self.read_image(path)
            return self.resize_image(img)
        except Exception as e:
            print(f"Error processing image {path}: {e}")
            return None

    def process_batch(self, batch):
        # print(type(batch))
        # print(batch)
        # return None
        return [self.process_func(path) for path in batch]

    def process_batch_with_key(self, batch, key: str):
        if key in batch:
            batch = batch[key]
        return {key: self.process_batch(batch)}

    def configure_processor(self, *args, **kwargs) -> Callable:
        return partial(self.process_batch_with_key, *args, **kwargs)


# @staticmethod
# def aspect_ratio(img: torch.Tensor):

#     minside = np.min(img.shape[1:])
#     maxside = np.max(img.shape[1:])

#     aspect_ratio = maxside / minside
#     return aspect_ratio
