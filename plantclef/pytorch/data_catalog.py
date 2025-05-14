"""
file: data_catalog.py
Created on Tuesday May 13th, 2025
Created by: Jacob A Rose


"""

from dataclasses import dataclass


# data_subset_paths = {
#     "train": "/teamspace/studios/this_studio/plantclef-vision/data/hf/plantclef2025/single_label_train_val_test/shortest_edge_308/train",
#     "val": "/teamspace/studios/this_studio/plantclef-vision/data/hf/plantclef2025/single_label_train_val_test/shortest_edge_308/val",
#     "test": "/teamspace/studios/this_studio/plantclef-vision/data/hf/plantclef2025/single_label_train_val_test/shortest_edge_308/test",
# }


@dataclass
class SubsetDataConfig:
    train: str
    val: str
    test: str


@dataclass
class DatasetConfig:
    subset_paths: SubsetDataConfig
    x_col: str
    y_col: str


PlantCLEF2024DatasetConfig = DatasetConfig(
    subset_paths=SubsetDataConfig(
        **{
            "train": "/teamspace/studios/this_studio/plantclef-vision/data/hf/plantclef2025/single_label_train_val_test/shortest_edge_308/train",
            "val": "/teamspace/studios/this_studio/plantclef-vision/data/hf/plantclef2025/single_label_train_val_test/shortest_edge_308/val",
            "test": "/teamspace/studios/this_studio/plantclef-vision/data/hf/plantclef2025/single_label_train_val_test/shortest_edge_308/test",
        }
    ),
    x_col="image",
    y_col="label_idx",
)


def make_dataset_from_config(cfg: DatasetConfig, **kwargs):
    """
    Create a dataset from the configuration.
    """
    from plantclef.pytorch.data import HFPlantDatasetDict

    return HFPlantDatasetDict(**PlantCLEF2024DatasetConfig(), **kwargs)
