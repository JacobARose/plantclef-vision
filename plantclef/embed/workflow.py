"""
file: plantclef/embed/workflow.py
Created around: April/May, 2025
Created by: Jacob A Rose


If you run this script, it will create embeddings and logits for the full unlabeled quadrat test set using a 3x3 tile grid input to the DINOv2 model and save them to disk.


python /teamspace/studios/this_studio/plantclef-vision/plantclef/embed/workflow.py --subsets "val" --batch_size 256


python /teamspace/studios/this_studio/plantclef-vision/plantclef/embed/workflow.py --subsets "test" --batch_size 512 --image_size 308 --tail 15360



python /teamspace/studios/this_studio/plantclef-vision/plantclef/embed/workflow.py --subsets "val" --batch_size 512 --pin_memory --non_blocking --prefetch_factor 4



python /teamspace/studios/this_studio/plantclef-vision/plantclef/embed/workflow.py --subsets "val" --batch_size 512 --non_blocking --prefetch_factor 2 --image_size 308


python /teamspace/studios/this_studio/plantclef-vision/plantclef/embed/workflow.py --subsets "val" --batch_size 512 --pin_memory --prefetch_factor 2 --image_size 308



python /teamspace/studios/this_studio/plantclef-vision/plantclef/embed/workflow.py --subsets "val" --batch_size 256 --non_blocking --prefetch_factor 4 --image_size 308

"""

import argparse
from dataclasses import dataclass, field
from typing import Optional, List, Dict
import torch
import os
import pandas as pd
from plantclef.pytorch.data import (
    HFPlantDataset,
)
from plantclef.utils.df_utils import save_df_to_parquet
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
    model: Optional[DINOv2LightningModel] = None,
    batch_size: int = 32,
    use_grid: bool = False,
    grid_size: int = 1,
    cpu_count: int = 1,
    top_k: int = 5,
    device: Optional[str] = None,
    pin_memory: bool = False,
    non_blocking: bool = False,
    prefetch_factor: int = 2,
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
    model = model or DINOv2LightningModel(top_k=top_k, non_blocking=non_blocking)

    # set device to GPU if available, otherwise CPU
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = pin_memory if device == "cuda" else False

    model.to(device)

    # create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=cpu_count,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
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
    i = 0
    try:
        for batch in tqdm(
            dataloader, desc="Extracting embeddings and logits", unit="batch"
        ):
            try:
                if isinstance(batch, dict):
                    batch = batch[x_col]
                embeddings, logits = predict_step(batch, batch_idx=0)
                embeddings = embeddings.cpu()
                all_embeddings.append(embeddings)
                # Each image in the batch gets a list of grid_size**2 dicts, each containing the top-k logits for that grid tile
                all_logits.extend(logits)
                i += 1
            except Exception as e:
                import ipdb

                ipdb.set_trace()
                print(f"Error during batch processing: {e}")

        embeddings = torch.cat(
            all_embeddings, dim=0
        )  # shape: [len(df), grid_size**2, 768]
    except Exception as e:
        import ipdb

        ipdb.set_trace()
        print(f"Error during inference: {e}")

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


def format_predictions_df(df):
    """
    Format the predictions DataFrame by expanding the top-k predictions and logits into separate columns.

    This is necessary for going
        from a single "logits" column, containing a dictionary of {prediction : logit} pairs
        to k*2 columns, \forall k in [0, 1, ..., i, ..., k-2, k-1], there's a f"pred_{i}" and a f"logit_{i}" column
    """

    def expand_logits_fn(row):
        logits = row["logits"]
        logits = [(k, v) for k, v in logits.items() if v is not None]
        logits = sorted(logits, key=lambda x: x[1], reverse=True)
        new_cols = {}  # "image_name": row.index}
        for i, (k, v) in enumerate(logits):
            new_cols[f"pred_{i}"] = k
            new_cols[f"logit_{i}"] = v
        return pd.Series(new_cols)

    df = df.set_index("image_name")
    df = df.apply(expand_logits_fn, result_type="expand", axis=1).reset_index()

    df = df.convert_dtypes()

    return df


def create_predictions_df(
    ds: HFPlantDataset, logits: list, group_tiles_by_image_name: bool = False
) -> pd.DataFrame:
    """
    Args:
        ds: HFPlantDataset
        embeddings: torch.Tensor
        logits: list
        group_tiles_by_image_name: bool, if True, will group tiles by image name. Only use if each image has been split into tiles, and predictions made individually on each tile.
            * [TODO] -- Add logic to automatically determine if there are > 1 prediction per image, indicating the need to have this value be True.


    Accepts an HFPlantDataset and a set of embeddings and logits.
    Create a DataFrame with predictions and save it to a CSV file.
    """
    # Create a DataFrame with image names and logits
    img_df = pd.DataFrame({"image_name": ds.dataset["image_name"]})  # type: ignore

    img_df = img_df.convert_dtypes()
    pred_df = img_df.assign(logits=logits)

    pred_df = format_predictions_df(pred_df)

    if group_tiles_by_image_name:
        pred_df = pred_df.assign(tile=pred_df.groupby("image_name").cumcount())

    return pred_df


def create_embeddings_df(
    ds, embeddings: "torch.Tensor", group_tiles_by_image_name: bool = False
) -> pd.DataFrame:
    """
    Create a DataFrame with embeddings and save it to a CSV file.

    """
    img_df = pd.DataFrame({"image_name": ds.dataset["image_name"]})

    # img_df = img_df.convert_dtypes()
    embeddings_df = img_df.assign(embeddings=embeddings.cpu().tolist())
    embeddings_df = embeddings_df.explode(["embeddings"], ignore_index=True)
    embeddings_df = embeddings_df.assign(
        emb_idx=embeddings_df.groupby("image_name").cumcount()
    )
    embeddings_df = embeddings_df.pivot(
        index="image_name", columns="emb_idx", values="embeddings"
    )
    embeddings_df = embeddings_df.reset_index()
    embeddings_df = embeddings_df.convert_dtypes()

    if group_tiles_by_image_name:
        embeddings_df = embeddings_df.assign(
            tile=embeddings_df.groupby("image_name").cumcount()
        )

    return embeddings_df


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
    head: Optional[int] = None
    tail: Optional[int] = None

    pin_memory: bool = False
    non_blocking: bool = False
    prefetch_factor: int = 2

    root_dir: str = (
        "/teamspace/studios/this_studio/plantclef-vision/data/plantclef2025/processed"
    )
    subset_embeddings_paths: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self, *args, **kwargs):
        self.dataset_processed_dir = f"{self.root_dir}/{self.dataset_name}"
        self.subset_embeddings_paths = {
            subset: f"{self.dataset_processed_dir}/{subset}/embeddings.parquet"
            for subset in self.subsets
        }
        self.subset_predictions_paths = {
            subset: f"{self.dataset_processed_dir}/{subset}/predictions.parquet"
            for subset in self.subsets
        }
        self.config_path = f"{self.dataset_processed_dir}/{self.subsets}-config.json"

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
            "--root_dir",
            type=str,
            default="/teamspace/studios/this_studio/plantclef-vision/data/plantclef2025/processed",
            help="Root directory for saving embeddings + logits",
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="If set, will print config and exit without running the pipeline",
        )
        parser.add_argument(
            "--head",
            type=int,
            help="[Optional] -- If set, will only run the pipeline on the first N images in the dataset",
        )
        parser.add_argument(
            "--tail",
            type=int,
            help="[Optional] -- If set, will only run the pipeline on the last N images in the dataset",
        )
        parser.add_argument(
            "--pin_memory",
            action="store_true",
            help="Enable pin_memory for DataLoader",
        )
        parser.add_argument(
            "--non_blocking",
            action="store_true",
            help="Enable non_blocking for DataLoader",
        )
        parser.add_argument(
            "--prefetch_factor",
            type=int,
            default=2,
            help="Number of batches to prefetch for DataLoader",
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
    ds = make_dataset(name="plantclef2024", load_all_subsets=False, subset="val")

    model = DINOv2LightningModel(top_k=cfg.top_k)

    for subset in cfg.subsets:
        ds.set_subset(subset)
        ds.set_transform(crop_size=cfg.image_size)

        num_samples = len(ds)

        if cfg.head:
            ds.dataset = ds.dataset.take(cfg.head)  # type: ignore
            print(
                f"[HEAD] Running on the first cfg.head = {cfg.head} images in the dataset",
                f"Dataset is now len(ds) = {len(ds)}",
            )
        if cfg.tail:
            ds.dataset = ds.dataset.add_column("idx", list(range(num_samples)))  # type: ignore
            ds.dataset = ds.dataset.select(  # type: ignore
                list(range((num_samples - cfg.tail), num_samples))
            )  # type: ignore
            # ds.dataset = ds.dataset[-cfg.tail :]  # type: ignore
            print(
                f"[TAIL] Running on the last cfg.tail = {cfg.tail} images in the dataset",
                f"Dataset is now len(ds) = {len(ds)}",
            )

        subset_embeddings_path = cfg.subset_embeddings_paths[subset]
        subset_predictions_path = cfg.subset_predictions_paths[subset]

        print(
            f"Initiating torch_pipeline on dataset subset {subset} of <<length {len(ds)}>> using <<batch_size {cfg.batch_size}>>"
        )

        print_current_time()
        try:
            embeddings, logits = torch_pipeline(
                dataset=ds,
                model=model,
                batch_size=cfg.batch_size,
                use_grid=cfg.use_grid,
                grid_size=cfg.grid_size,
                cpu_count=cfg.cpu_count,
                top_k=cfg.top_k,
                device=cfg.device,
                pin_memory=cfg.pin_memory,
                non_blocking=cfg.non_blocking,
                prefetch_factor=cfg.prefetch_factor,
            )
        except Exception as e:
            print(f"Error during torch_pipeline: {e}")
            import ipdb

            ipdb.set_trace()

        try:
            pred_df = create_predictions_df(
                ds, logits=logits, group_tiles_by_image_name=cfg.use_grid
            )
            save_df_to_parquet(
                pred_df, path=subset_predictions_path, max_rows_per_partition=10000
            )
            print(f"Predictions saved to {subset_predictions_path}")
            print_current_time()
            print_dir_size(subset_predictions_path)
        except Exception as e:
            print(f"Error during processing and saving of predictions: {e}")
            import ipdb

            ipdb.set_trace()

        try:
            embed_df = create_embeddings_df(
                ds, embeddings=embeddings, group_tiles_by_image_name=cfg.use_grid
            )
            save_df_to_parquet(
                embed_df, path=subset_embeddings_path, max_rows_per_partition=10000
            )
            print(f"Predictions saved to {subset_embeddings_path}")
            print_current_time()
            print_dir_size(subset_embeddings_path)
        except Exception as e:
            print(f"Error during processing and saving of embeddings: {e}")
            import ipdb

            ipdb.set_trace()

        del logits
        del embeddings


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

#     # root_dir: str = ""
#     # root_dir: str = ""
#     # root_dir: str = ""
#     # test_embeddings_dir: str = ""
#     # folder_name: str = ""
#     # test_embeddings_path: str = ""
#     # test_submission_path: str = ""
#     # config_path: str = ""

#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.root_dir = f"{self.root_dir}/embeddings"
#         self.test_embeddings_dir = f"{self.root_dir}/full_test"
#         self.root_dir = f"{self.root_dir}/embeddings"
#         self.test_embeddings_dir = f"{self.root_dir}/full_test"
#         self.root_dir = f"{self.root_dir}/embeddings"
#         self.test_embeddings_dir = f"{self.root_dir}/full_test"
#         self.folder_name = f"test_grid_{self.grid_size}x{self.grid_size}_embeddings"
#         self.test_embeddings_path = f"{self.test_embeddings_dir}/{self.folder_name}"
#         self.test_submission_path = (
#             f"{self.test_embeddings_dir}/{self.folder_name}-submission.csv"
#         )
#         self.config_path = f"{self.test_embeddings_path}-config.json"
