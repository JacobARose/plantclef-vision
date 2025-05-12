from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import os
import pandas as pd
from typing import Optional, Dict
import torch
import pytorch_lightning as pl

from functools import partial
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as PyTorchDataset
from torchvision.transforms import ToTensor
from torchvision import transforms
from datasets import Dataset as HFDataset, concatenate_datasets
import PIL.Image
from plantclef.serde import deserialize_image
from plantclef.pytorch.model import DINOv2LightningModel
from tqdm import tqdm


def custom_collate_fn(batch, use_grid):
    """Custom collate function to handle batched grid images properly."""
    if use_grid:
        return torch.stack(batch, dim=0)  # shape: (B, grid_size**2, C, H, W)
    return torch.stack(batch)  # shape: (B, C, H, W)


def custom_collate_fn_partial(use_grid):
    """Returns a pickle-friendly collate function with the `use_grid` flag."""
    return partial(custom_collate_fn, use_grid=use_grid)


class BasePlantDataset(ABC, PyTorchDataset):
    """Abstract base class for plant datasets."""

    def __init__(
        self,
        transform: Optional[transforms.Compose] = None,
        col_name: str = "data",
        use_grid: bool = False,
        grid_size: int = 4,
    ):
        """
        Args:
            transform: torchvision transforms.Compose object
            col_name: Column name containing image data
            use_grid: Whether to split images into a grid
            grid_size: Size of the grid to split images into
        """
        self.transform = transform or (lambda x: x)
        self.col_name = col_name
        self.use_grid = use_grid
        self.grid_size = grid_size

    @abstractmethod
    def _load_data(self) -> None:
        """Abstract method to load the dataset."""
        pass

    # def _split_into_grid(self, image) -> list:
    #     """Split an image into a grid of sub-images."""
    #     image = transforms.ToPILImage()(image)
    #     w, h = image.size
    #     grid_w, grid_h = w // self.grid_size, h // self.grid_size
    #     images = []
    #     for i in range(self.grid_size):
    #         for j in range(self.grid_size):
    #             left = i * grid_w
    #             upper = j * grid_h
    #             right = left + grid_w
    #             lower = upper + grid_h
    #             crop_image = image.crop((left, upper, right, lower))
    #             images.append(ToTensor()(crop_image))
    #     return images

    def _split_into_grid(
        self, image: torch.Tensor, grid_size: Optional[int] = None
    ) -> list:
        grid_size = grid_size or self.grid_size
        c, h, w = image.shape  # Extract height, width, and channels
        grid_h = h // grid_size
        grid_w = w // grid_size  # Tile size

        tiles = []
        for i in range(grid_size):
            for j in range(grid_size):
                top = i * grid_h
                left = j * grid_w
                bottom = top + grid_h
                right = left + grid_w
                tile = image[:, top:bottom, left:right]  # Slice the torch.Tensor
                tiles.append(tile)

        return tiles  # Returns a list of torch.Tensors

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Get an item from the dataset using memory-efficient loading."""

        img = self._get_image_tensor(idx)

        # Apply transforms to full image
        if self.transform:
            img = self.transform(img)

        # Split into grid if required
        if self.use_grid:
            img_list = self._split_into_grid(img)
            return torch.stack(img_list)
        else:
            return img

    @abstractmethod
    def _get_image_tensor(self, idx: int) -> torch.Tensor:
        """Abstract method to get image tensor from the dataset."""
        pass

    @abstractmethod
    def _get_image_bytes(self, idx: int) -> bytes:
        """Abstract method to get image bytes from the dataset."""
        pass

    def get_transforms(
        self,
        image_size: Optional[Dict[str, int]] = None,
        crop_size: Optional[Dict[str, int]] = None,
    ):
        image_size = image_size or {"shortest_edge": 588}
        crop_size = crop_size or {"height": 588, "width": 588}
        return transforms.Compose(
            [
                transforms.Resize(
                    size=image_size["shortest_edge"],
                    interpolation=transforms.InterpolationMode.BICUBIC,
                    max_size=None,
                    antialias=True,
                ),
                transforms.CenterCrop(size=(crop_size["height"], crop_size["width"])),
                transforms.Normalize(
                    mean=[0.4850, 0.4560, 0.4060], std=[0.2290, 0.2240, 0.2250]
                ),
            ]
        )

    @classmethod
    def center_crop(cls, image: torch.Tensor) -> torch.Tensor:
        min_dim = min(image.shape[1:])
        return transforms.CenterCrop(min_dim)(image)

    def plot_image_tiles(
        self,
        idx: int,
        grid_size: Optional[int] = None,
        figsize: tuple = (15, 8),
        dpi: int = 100,
        skip_transforms: bool = False,
        normalize_pixel_values: bool = True,
    ):
        """
        Display an original image and its tiles in a single figure.


        :param idx:
        :param grid_size: Number of tiles per row/column (grid_size x grid_size).
        :param figsize: Figure size (width, height).
        :param dpi: Dots per inch (image resolution).


        [] [TODO] -- Add option to include axis labels with x & y resolution visible on 1 of the grid_size x grid_size tiles
        """

        image = self._get_image_tensor(idx)
        if not skip_transforms:
            image = self.transform(image)

        image = self.center_crop(image)  # Ensure a square crop

        # Scale pixel values to be within 0.0 and 1.0
        max_val = torch.max(torch.abs(image))
        image = (image / max_val + 1.0) / 2.0

        # split image into tiles
        image_tiles = self._split_into_grid(image, grid_size)

        image_name = ""  # TBD
        grid_size = grid_size or self.grid_size

        # create figure with 1 row and 2 columns (original image | 3x3 grid)
        fig, axes = plt.subplots(1, 2, figsize=figsize, dpi=dpi)

        # Set main titles for left and right plots
        fig.text(0.25, 0.97, "Image", fontsize=20, fontweight="bold", ha="center")
        fig.text(0.75, 0.97, "Image Tiles", fontsize=20, fontweight="bold", ha="center")

        # plot original image on the left
        axes_left = axes[0]
        axes_left.imshow(image.permute(1, 2, 0))
        axes_left.set_title(image_name, fontsize=20)
        axes_left.set_xticks([])
        axes_left.set_yticks([])
        spines = ["top", "right", "bottom", "left"]
        for spine in spines:
            axes_left.spines[spine].set_visible(False)

        # plot the grid of tiles on the right
        gs = fig.add_gridspec(1, 2)[1]
        grid_axes = gs.subgridspec(grid_size, grid_size)

        if type(image_tiles) is torch.Tensor:
            assert (
                image_tiles.shape[0] == grid_size**2
            ), f"Expected {grid_size**2} tiles, got {image_tiles.shape[0]}"
            image_tiles = torch.split(image_tiles, 1, dim=0)

        for idx, tile in enumerate(image_tiles):
            row, col = divmod(idx, grid_size)
            ax = fig.add_subplot(grid_axes[row, col])
            ax.imshow(tile.permute(1, 2, 0))
            ax.set_xlabel(f"Tile {idx+1}", fontsize=15)
            ax.set_xticks([])
            ax.set_yticks([])
            for s in spines:
                ax.spines[s].set_visible(False)

        ax_right = axes[1]
        ax_right.set_title(f"{grid_size}x{grid_size} grid of tiles", fontsize=20)
        ax_right.set_xticks([])
        ax_right.set_yticks([])
        ax_right.axis("off")

        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.show()


class DataFramePlantDataset(BasePlantDataset):
    """Plant dataset implementation using a pandas DataFrame."""

    def __init__(
        self,
        df: pd.DataFrame,
        transform: Optional[transforms.Compose] = None,
        col_name: str = "data",
        use_grid: bool = False,
        grid_size: int = 4,
    ):
        """
        Args:
            df: Pandas DataFrame containing image binary data
            transform: torchvision transforms.Compose object
            col_name: Column name containing image data
            use_grid: Whether to split images into a grid
            grid_size: Size of the grid to split images into
        """
        super().__init__(transform, col_name, use_grid, grid_size)
        self.df = df

    def _load_data(self) -> None:
        """Load data from DataFrame."""
        self.df = self.df.copy()

    def _get_image_bytes(self, idx: int) -> bytes:
        """Get image bytes from DataFrame."""
        return self.df.iloc[idx][self.col_name]

    def _get_image_tensor(self, idx: int) -> torch.Tensor:
        example = self.df.iloc[idx, :]
        img_bytes = example[self.col_name]

        # Convert bytes to PIL image
        img = deserialize_image(img_bytes)
        return ToTensor()(img)

    def __len__(self) -> int:
        """Get the length of the dataset."""
        return len(self.df)


# print("about to define HFPlantDataset")
class HFPlantDataset(BasePlantDataset):
    """Plant dataset implementation using HuggingFace Dataset.load_from_disk."""

    def __init__(
        self,
        path: str,
        transform: Optional[transforms.Compose] = None,
        col_name: str = "data",
        use_grid: bool = False,
        grid_size: int = 4,
    ):
        """
        Args:
            path: Path to the saved dataset on disk
            transform: torchvision transforms.Compose object
            col_name: Column name containing image data
            use_grid: Whether to split images into a grid
            grid_size: Size of the grid to split images into
        """
        super().__init__(transform, col_name, use_grid, grid_size)
        self.path = path
        self.dataset = self._load_data(path)

    def _load_nested_data(self, path: str) -> HFDataset:
        """
        Load a directory of HuggingFace datasets and concatenate them.
        """
        contents = os.listdir(path)
        if "state.json" in contents and "dataset_info.json" in contents:
            return self._load_data(path)

        dataset_paths = [
            os.path.join(path, p) for p in contents if p.endswith(".arrow")
        ]
        return concatenate_datasets([self._load_data(p) for p in tqdm(dataset_paths)])

    def _load_data(self, path: str) -> HFDataset:
        """Load data from disk using HuggingFace Dataset."""
        return HFDataset.load_from_disk(path)

    def _get_image_pil(self, idx: int) -> PIL.Image.Image:
        """Get image as PIL.Image from HuggingFace Dataset."""
        example = self.dataset[idx]
        pil_img = example[self.col_name]
        return pil_img

    def _get_image_tensor(self, idx: int) -> torch.Tensor:
        example = self.dataset[idx]
        pil_img = example[self.col_name]
        return ToTensor()(pil_img)

    def _get_image_bytes(self, idx: int) -> bytes:
        pass

    #     """Get image bytes from HuggingFace Dataset."""
    #     return self.df.iloc[idx][self.col_name]

    def __len__(self) -> int:
        """Get the length of the dataset."""
        return len(self.dataset)

    def __repr__(self) -> str:
        """String representation of the dataset."""
        return f"HFPlantDataset(\n{self.dataset.__repr__()}\n)"


class PlantDataset(PyTorchDataset):
    """Custom PyTorch Dataset for loading plant images from a Pandas DataFrame."""

    def __init__(
        self,
        df,
        transform=None,
        col_name: str = "data",
        use_grid: bool = False,
        grid_size: int = 4,
    ):
        """
        Args:
            df (pd.DataFrame): Pandas DataFrame containing image binary data.
            transform (torchvision.transforms.Compose): Image transformations.
            use_grid (bool): Whether to split images into a grid.
            grid_size (int): The size of the grid to split images into.
        """
        self.df = df
        self.transform = transform
        self.col_name = col_name
        self.use_grid = use_grid
        self.grid_size = grid_size

    def __len__(self):
        return len(self.df)

    def _split_into_grid(self, image):
        w, h = image.size
        grid_w, grid_h = w // self.grid_size, h // self.grid_size
        images = []
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                left = i * grid_w
                upper = j * grid_h
                right = left + grid_w
                lower = upper + grid_h
                crop_image = image.crop((left, upper, right, lower))
                images.append(crop_image)
        return images

    def __getitem__(self, idx) -> list:
        img_bytes = self.df.iloc[idx][self.col_name]  # column with image bytes
        img = deserialize_image(img_bytes)  # convert from bytes to PIL image

        if self.use_grid:
            img_list = self._split_into_grid(img)
            if self.transform:
                img_list = [self.transform(image) for image in img_list]
            else:  # no transform, shape: (grid_size**2, C, H, W)
                img_list = [ToTensor()(image) for image in img_list]
            return torch.stack(img_list)
        # single image, shape: (C, H, W)
        if self.transform:
            return self.transform(img)  # (C, H, W)
        return ToTensor()(img)  # (C, H, W)


class PlantDataModule(pl.LightningDataModule):
    """LightningDataModule for handling dataset loading and preparation."""

    def __init__(
        self,
        pandas_df,
        batch_size=32,
        use_grid=False,
        grid_size=4,
        num_workers=4,
    ):
        super().__init__()
        self.pandas_df = pandas_df
        self.batch_size = batch_size
        self.use_grid = use_grid
        self.grid_size = grid_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        """Set up dataset and transformations."""

        self.model = DINOv2LightningModel()
        self.dataset = PlantDataset(
            self.pandas_df,
            self.model.transform,  # Use the model's transform
            use_grid=self.use_grid,
            grid_size=self.grid_size,
        )

    def predict_dataloader(self):
        """Returns DataLoader for inference."""
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True,
            collate_fn=custom_collate_fn_partial(self.use_grid),
        )
