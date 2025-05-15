"""
file: plantclef/embed/workflow.py
Created around: April/May, 2025
Created by: Jacob A Rose


If you run this script, it will create embeddings and logits for the full unlabeled quadrat test set using a 3x3 tile grid input to the DINOv2 model and save them to disk.



"""

import argparse
from dataclasses import dataclass, field
from typing import Optional, List, Dict
import torch
import os
import pandas as pd
from plantclef.pytorch.data import (
    HFDataset,
    HFPlantDataset,
)
from plantclef.pytorch.data_catalog import make_dataset
from plantclef.pytorch.model import DINOv2LightningModel
from plantclef.embed.utils import print_current_time, print_dir_size
from plantclef.config import BaseConfig
from rich.repr import auto
from rich import print as pprint
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb


def torch_pipeline(
    dataset: HFPlantDataset,
    batch_size: int = 32,
    use_grid: bool = False,
    grid_size: int = 1,
    cpu_count: int = 1,
    top_k: int = 5,
    device: Optional[str] = None,
):
    """
    Pipeline to extract embeddings and top-K logits using PyTorch Lightning.
    :param dataset: Dataset to use for inference.
    :param batch_size: Size of the batches for DataLoader.
    :param use_grid: Boolean indicating whether to use grid processing.
    :param grid_size: Number of grid cells to split each image into if use_grid is True.
    :param cpu_count: Number of CPU cores to use for DataLoader.
    :param top_k: Number of top logits to extract.
    :param device: Device to use for inference. If None, will use GPU if available, otherwise CPU.
    """

    # initialize model
    model = DINOv2LightningModel(top_k=top_k)

    # set device to GPU if available, otherwise CPU
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=cpu_count,
        pin_memory=True,
        # collate_fn=custom_collate_fn_partial(use_grid),
    )
    x_col = dataset.x_col
    if use_grid:
        predict_step = model.predict_grid_step
    else:
        predict_step = model.predict_step

    # run inference and collect embeddings with tqdm progress bar
    all_embeddings = []
    all_logits = []
    for batch in tqdm(
        dataloader, desc="Extracting embeddings and logits", unit="batch"
    ):
        if isinstance(batch, dict):
            batch = batch[x_col]
        embeddings, logits = predict_step(batch, batch_idx=0)
        all_embeddings.append(embeddings.cpu())
        # Each image in the batch gets a list of grid_size**2 dicts, each containing the top-k logits for that grid tile
        all_logits.extend(logits)

    embeddings = torch.cat(all_embeddings, dim=0)  # shape: [len(df), grid_size**2, 768]

    return embeddings, all_logits


# def pl_trainer_pipeline(
#     pandas_df: pd.DataFrame,
#     batch_size: int = 32,
#     use_grid: bool = False,
#     grid_size: int = 1,
#     cpu_count: int = 1,
#     top_k: int = 5,
# ):
#     """Pipeline to extract embeddings and top-k logits using PyTorch Lightning."""

#     # initialize DataModule
#     # data_module = PlantDataModule(
#     #     pandas_df,
#     #     batch_size=batch_size,
#     #     use_grid=use_grid,
#     #     grid_size=grid_size,
#     #     num_workers=cpu_count,
#     # )

#     # initialize Model
#     model = DINOv2LightningModel(top_k=top_k)

#     # define Trainer (inference mode)
#     trainer = pl.Trainer(
#         accelerator=get_device(),
#         devices=1,
#         enable_progress_bar=True,
#     )

#     # run Inference
#     predictions = trainer.predict(model, datamodule=data_module) or []

#     all_embeddings = []
#     all_logits = []
#     for batch in predictions:
#         embed_batch, logits_batch = batch  # batch: List[Tuple[embeddings, logits]]
#         all_embeddings.append(embed_batch)  # keep embeddings as tensors
#         reshaped_logits = [
#             logits_batch[i : i + grid_size**2]
#             for i in range(0, len(logits_batch), grid_size**2)
#         ]
#         all_logits.extend(reshaped_logits)  # preserve batch structure

#     # convert embeddings to tensor
#     embeddings = torch.cat(all_embeddings, dim=0)  # shape: [len(df), grid_size**2, 768]

#     if use_grid:
#         embeddings = embeddings.view(-1, grid_size**2, 768)
#     else:
#         embeddings = embeddings.view(-1, 1, 768)

#     return embeddings, all_logits


def create_predictions_df(
    ds: HFPlantDataset, embeddings: torch.Tensor, logits: list
) -> pd.DataFrame:
    """
    Accepts an HFPlantDataset and a set of embeddings and logits.

    To be called after the model has been run on the full dataset in ds.

    Returns a DataFrame with the following columns:
        - image_name
        - tile
        - embeddings
        - logits
    The DataFrame is exploded to have one row per tile.

    """

    pred_df = pd.DataFrame({"image_name": ds.dataset["image_name"]})
    # pred_df["image_name"] = pred_df["image_name"].str.rsplit("/", n=1, expand=True)[1]

    pred_df = pred_df.convert_dtypes()

    pred_df = pred_df.assign(embeddings=embeddings.cpu().tolist(), logits=logits)
    explode_df = pred_df.explode(["embeddings", "logits"], ignore_index=True)
    explode_df = explode_df.assign(tile=explode_df.groupby("image_name").cumcount())

    return explode_df


@dataclass
@auto
class PipelineConfig(BaseConfig):
    """
    Configuration class for the embedding pipeline.

    """

    dataset_name: str = "plantclef2024"
    subsets: List[str] = field(default_factory=lambda: ["train", "val", "test"])
    use_grid: bool = False
    grid_size: int = 3
    image_size: int = 518
    batch_size: int = 32
    cpu_count: int = os.cpu_count() or 1
    top_k: int = 5
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    embeddings_root_dir: str = (
        "/teamspace/studios/this_studio/plantclef-vision/data/plantclef2025/embeddings"
    )
    subset_embeddings_paths: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self, *args, **kwargs):
        self.dataset_embeddings_dir = f"{self.embeddings_root_dir}/{self.dataset_name}"
        self.subset_embeddings_paths = {
            subset: f"{self.dataset_embeddings_dir}/{subset}" for subset in self.subsets
        }
        self.config_path = f"{self.dataset_embeddings_dir}/{self.subsets}-config.json"

    @staticmethod
    def parse_args() -> argparse.Namespace:
        """
        Parse command-line arguments to override default PipelineConfig values.
        """

        parser = argparse.ArgumentParser(description="PipelineConfig Arguments")
        parser.add_argument(
            "--dataset_name",
            type=str,
            default="plantclef2024",
            help="Name of the dataset",
        )
        parser.add_argument(
            "--subsets",
            type=str,
            nargs="+",
            default=["train", "val", "test"],
            help="Dataset subsets",
        )
        parser.add_argument(
            "--use_grid", action="store_true", help="Enable grid processing"
        )
        parser.add_argument(
            "--grid_size", type=int, default=3, help="Grid size for image tiling"
        )
        parser.add_argument(
            "--image_size", type=int, default=518, help="Image size for processing"
        )
        parser.add_argument(
            "--batch_size", type=int, default=32, help="Batch size for DataLoader"
        )
        parser.add_argument(
            "--cpu_count",
            type=int,
            default=os.cpu_count() or 1,
            help="Number of CPU cores to use",
        )
        parser.add_argument(
            "--top_k", type=int, default=5, help="Number of top logits to extract"
        )
        parser.add_argument(
            "--device",
            type=str,
            default="cuda" if torch.cuda.is_available() else "cpu",
            help="Device to use for inference",
        )
        parser.add_argument(
            "--embeddings_root_dir",
            type=str,
            default="/teamspace/studios/this_studio/plantclef-vision/data/plantclef2025/embeddings",
            help="Root directory for saving embeddings + logits",
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="If set, will print config and exit without running the pipeline",
        )

        args = parser.parse_args()

        # If dry-run is set, print the config and exit

        return args

    @classmethod
    def from_args(cls, args: Optional[Dict | argparse.Namespace] = {}):
        """
        Create a PipelineConfig instance from command-line arguments.
        """
        args = vars((args or cls.parse_args()))
        dry_run = args.pop("dry_run")
        config = cls(**(args))

        if dry_run:
            print("[DRY RUN] Args:")
            pprint(args)
            print("[DRY RUN] Configuration:")
            pprint(config)
            exit(0)

        print("[FULL RUN] Args:")
        pprint(args)
        print("[FULL RUN] Configuration:")
        pprint(config)
        return config


def embed_predict_save(
    cfg: PipelineConfig,
):
    """
    Make predictions and save them to disk.
    """

    # ds = HFPlantDataset(
    #     path=cfg.hf_dataset_dir,
    #     transform=None,
    #     col_name="image",
    #     use_grid=cfg.use_grid,
    #     grid_size=cfg.grid_size,
    # )
    # ds.transform = ds.get_transforms(cfg.image_size)

    print(f"[RUNNING] make_dataset(name={cfg.dataset_name}, load_all_subsets=False)")
    ds = make_dataset(name="plantclef2024", load_all_subsets=False)

    for subset in cfg.subsets:
        ds.set_subset(subset)
        ds.set_transform(crop_size=cfg.image_size)

        subset_embeddings_path = cfg.subset_embeddings_paths[subset]
        # Create the directory if it doesn't exist
        print(f"[RUNNING] os.makedirs({subset_embeddings_path}, exist_ok=True)")
        os.makedirs(subset_embeddings_path, exist_ok=True)

        print(
            f"Initiating torch_pipeline on dataset subset {subset} of length {len(ds)} using batch_size {cfg.batch_size}"
        )

        print_current_time()

        embeddings, logits = torch_pipeline(
            dataset=ds,
            batch_size=cfg.batch_size,
            use_grid=cfg.use_grid,
            grid_size=cfg.grid_size,
            cpu_count=cfg.cpu_count,
            top_k=cfg.top_k,
            device=cfg.device,
        )

        pred_df = create_predictions_df(ds, embeddings, logits)

        pred_ds = HFDataset.from_pandas(pred_df)
        pred_ds.save_to_disk(subset_embeddings_path)

        print(f"Predictions saved to {subset_embeddings_path}")
        print_current_time()

        print_dir_size(subset_embeddings_path)


def run_embed_test(args: Optional[argparse.Namespace] = None):
    """

    Create embeddings from the full test set using the DINOv2 model and save them to disk.

    """
    cfg = PipelineConfig.from_args(args)
    try:
        wandb.init(
            project="plantclef2024",
            entity="jrose",
            config=cfg.to_dict(),
            name=f"embed-{cfg.dataset_name}-{cfg.subsets}",
            job_type="embed",
            tags=["embed", *cfg.subsets],
        )
        embed_predict_save(cfg)
    except Exception as e:
        print(f"Error in embed_predict_save: {e}")
        raise e

    print("All done!")
    print_current_time()


if __name__ == "__main__":
    print(f"Initiating function run_embed_test() from file: {__file__}")
    print_current_time()

    run_embed_test()


#################

# [EDITOR'S NOTE] -- (Added Wednesday May 14th, 2025) -- The following is legacy code for just the platnclef2025 unlabeled test quadrat images, while I focus on making the plantclef2024 train-val-test subsets to work.


# @dataclass
# @auto
# class PipelineConfig(BaseConfig):
#     """
#     Configuration class for the embedding pipeline.

#     """
#     dataset_name: str = "plantclef2024"
#     subsets: List[str] = field(default_factory=lambda: ["train", "val", "test"])
#     use_grid: bool = False
#     grid_size: int = 3
#     image_size: int = 518
#     batch_size: int = 32
#     cpu_count: int = os.cpu_count() or 1
#     top_k: int = 5
#     device: str = "cuda" if torch.cuda.is_available() else "cpu"


#     # root_dir: str = "/teamspace/studios/this_studio/plantclef-vision/data/plantclef2025"
#     # dataset_dir: str = "/teamspace/studios/this_studio/plantclef-vision/data/plantclef2025/competition-metadata/PlantCLEF2025_test_images/PlantCLEF2025_test_images"
#     # hf_dataset_dir: str = "/teamspace/studios/this_studio/plantclef-vision/data/parquet/plantclef2025/full_test/HF_dataset"

#     # embeddings_root_dir: str = ""
#     # test_embeddings_dir: str = ""
#     # folder_name: str = ""
#     # test_embeddings_path: str = ""
#     # test_submission_path: str = ""
#     # config_path: str = ""

#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.embeddings_root_dir = f"{self.root_dir}/embeddings"
#         self.test_embeddings_dir = f"{self.embeddings_root_dir}/full_test"
#         self.folder_name = f"test_grid_{self.grid_size}x{self.grid_size}_embeddings"
#         self.test_embeddings_path = f"{self.test_embeddings_dir}/{self.folder_name}"
#         self.test_submission_path = (
#             f"{self.test_embeddings_dir}/{self.folder_name}-submission.csv"
#         )
#         self.config_path = f"{self.test_embeddings_path}-config.json"
