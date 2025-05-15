"""
file: data_catalog.py
Created on Tuesday May 13th, 2025
Created by: Jacob A Rose


Provides Config dataclasses for specific datasets, plus helper functions to load them into HFPlantDatasetDicts.


Run script to see available datasets and their configs.


"""

from dataclasses import dataclass, asdict
from plantclef.pytorch.data import HFPlantDatasetDict


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
    paths: SubsetDataConfig
    x_col: str
    y_col: str
    id_col: str


def get_config_from_dataset(ds: HFPlantDatasetDict) -> DatasetConfig:
    """
    Get the DatasetConfig from a HFPlantDatasetDict.
    """
    return DatasetConfig(
        paths=SubsetDataConfig(**ds.paths),
        x_col=ds.x_col,
        y_col=ds.y_col,
        id_col=ds.id_col,
    )


PlantCLEF2024DatasetConfig = DatasetConfig(
    paths=SubsetDataConfig(
        **{
            "train": "/teamspace/studios/this_studio/plantclef-vision/data/hf/plantclef2025/single_label_train_val_test/shortest_edge_308/train",
            "val": "/teamspace/studios/this_studio/plantclef-vision/data/hf/plantclef2025/single_label_train_val_test/shortest_edge_308/val",
            "test": "/teamspace/studios/this_studio/plantclef-vision/data/hf/plantclef2025/single_label_train_val_test/shortest_edge_308/test",
        }
    ),
    x_col="image",
    y_col="label_idx",
    id_col="image_name",
)


def make_dataset_from_config(cfg: DatasetConfig, **kwargs) -> HFPlantDatasetDict:
    """
    Create a dataset from the configuration.
    """
    from plantclef.pytorch.data import HFPlantDatasetDict

    return HFPlantDatasetDict(**asdict(cfg), **kwargs)


###################################################################
###################################################################

###################################################################
###################################################################


available_datasets_configs = {"plantclef2024": PlantCLEF2024DatasetConfig}


def make_dataset(name: str = "", **kwargs) -> HFPlantDatasetDict:
    """
    Main function to create an HFPlantDatasetDict using a known DatasetConfig by querying with a str name.
    """

    if name not in available_datasets_configs:
        raise ValueError(
            f"Dataset {name} not found in available datasets: {available_datasets_configs.keys()}"
        )
    cfg = available_datasets_configs[name]

    return make_dataset_from_config(cfg, **kwargs)


if __name__ == "__main__":
    from pprint import pprint

    print(f"Available datasets: {available_datasets_configs.keys()}")

    for name, cfg in available_datasets_configs.items():
        print(f"Dataset: {name}")
        pprint(cfg)
        print()
