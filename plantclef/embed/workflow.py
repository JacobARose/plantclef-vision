"""
file: plantclef/embed/workflow.py
Created around: April/May, 2025
Created by: Jacob A Rose


If you run this script, it will create embeddings and logits for the full unlabeled quadrat test set using a 3x3 tile grid input to the DINOv2 model and save them to disk.



"""

from dataclasses import dataclass
from typing import Optional
import torch
import os
import pandas as pd
import pytorch_lightning as pl
from plantclef.pytorch.data import (
    HFDataset,
    HFPlantDataset,
    PlantDataModule,
    custom_collate_fn_partial,
)
from plantclef.pytorch.model import DINOv2LightningModel
from plantclef.embed.utils import print_current_time, print_dir_size
from plantclef.config import get_device, BaseConfig
from rich.repr import auto
from torch.utils.data import DataLoader
from tqdm import tqdm


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
        collate_fn=custom_collate_fn_partial(use_grid),
    )

    # run inference and collect embeddings with tqdm progress bar
    all_embeddings = []
    all_logits = []
    for batch in tqdm(
        dataloader, desc="Extracting embeddings and logits", unit="batch"
    ):
        embeddings, logits = model.predict_grid_step(batch, batch_idx=0)
        all_embeddings.append(embeddings.cpu())
        # Each image in the batch gets a list of grid_size**2 dicts, each containing the top-k logits for that grid tile
        all_logits.extend(logits)

    embeddings = torch.cat(all_embeddings, dim=0)  # shape: [len(df), grid_size**2, 768]

    return embeddings, all_logits


def pl_trainer_pipeline(
    pandas_df: pd.DataFrame,
    batch_size: int = 32,
    use_grid: bool = False,
    grid_size: int = 1,
    cpu_count: int = 1,
    top_k: int = 5,
):
    """Pipeline to extract embeddings and top-k logits using PyTorch Lightning."""

    # initialize DataModule
    data_module = PlantDataModule(
        pandas_df,
        batch_size=batch_size,
        use_grid=use_grid,
        grid_size=grid_size,
        num_workers=cpu_count,
    )

    # initialize Model
    model = DINOv2LightningModel(top_k=top_k)

    # define Trainer (inference mode)
    trainer = pl.Trainer(
        accelerator=get_device(),
        devices=1,
        enable_progress_bar=True,
    )

    # run Inference
    predictions = trainer.predict(model, datamodule=data_module) or []

    all_embeddings = []
    all_logits = []
    for batch in predictions:
        embed_batch, logits_batch = batch  # batch: List[Tuple[embeddings, logits]]
        all_embeddings.append(embed_batch)  # keep embeddings as tensors
        reshaped_logits = [
            logits_batch[i : i + grid_size**2]
            for i in range(0, len(logits_batch), grid_size**2)
        ]
        all_logits.extend(reshaped_logits)  # preserve batch structure

    # convert embeddings to tensor
    embeddings = torch.cat(all_embeddings, dim=0)  # shape: [len(df), grid_size**2, 768]

    if use_grid:
        embeddings = embeddings.view(-1, grid_size**2, 768)
    else:
        embeddings = embeddings.view(-1, 1, 768)

    return embeddings, all_logits


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

    pred_df = pd.DataFrame({"image_name": ds.dataset["file_path"]})
    pred_df["image_name"] = pred_df["image_name"].str.rsplit("/", n=1, expand=True)[1]

    pred_df = pred_df.convert_dtypes()

    pred_df = pred_df.assign(embeddings=embeddings.cpu().tolist(), logits=logits)
    explode_df = pred_df.explode(["embeddings", "logits"], ignore_index=True)
    explode_df = explode_df.assign(tile=explode_df.groupby("image_name").cumcount())

    return explode_df


@dataclass
@auto
class Config(BaseConfig):
    use_grid: bool = True
    grid_size: int = 3
    image_size: int = 546
    batch_size: int = 16
    cpu_count: int = os.cpu_count() or 1
    top_k: int = 5
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    root_dir: str = "/teamspace/studios/this_studio/plantclef-vision/data/plantclef2025"
    dataset_dir: str = "/teamspace/studios/this_studio/plantclef-vision/data/plantclef2025/competition-metadata/PlantCLEF2025_test_images/PlantCLEF2025_test_images"
    hf_dataset_dir: str = "/teamspace/studios/this_studio/plantclef-vision/data/parquet/plantclef2025/full_test/HF_dataset"

    embeddings_dir: str = ""
    test_embeddings_dir: str = ""
    folder_name: str = ""
    test_embeddings_path: str = ""
    test_submission_path: str = ""
    config_path: str = ""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.embeddings_dir = f"{self.root_dir}/embeddings"
        self.test_embeddings_dir = f"{self.embeddings_dir}/full_test"
        self.folder_name = f"test_grid_{self.grid_size}x{self.grid_size}_embeddings"
        self.test_embeddings_path = f"{self.test_embeddings_dir}/{self.folder_name}"
        self.test_submission_path = (
            f"{self.test_embeddings_dir}/{self.folder_name}-submission.csv"
        )
        self.config_path = f"{self.test_embeddings_path}-config.json"

    def save(
        self,
        path: Optional[str] = None,
        indent: Optional[int] = None,
        exist_ok: bool = True,
    ) -> str:
        """
        Save the config parameters to a JSON file.

        If path is None, the config will be saved to the default path.

        To load, use `cfg = Config.load(path)`.

        Args:
            filename: Path to save the file
            indent: Optional indentation for pretty printing (None for compact)
        """

        path = path or self.config_path

        if not path.endswith(".json"):
            path = f"{path}.json"

        super().save(path, indent, exist_ok=exist_ok)
        self.config_path = path
        return self.config_path


def make_predictions_and_save(
    cfg: Config,
):
    """
    Make predictions and save them to disk.
    """

    # Create the directory if it doesn't exist
    print(f"[RUNNING] os.makedirs({cfg.test_embeddings_dir}, exist_ok=True)")
    os.makedirs(cfg.test_embeddings_dir, exist_ok=True)

    ds = HFPlantDataset(
        path=cfg.hf_dataset_dir,
        transform=None,
        col_name="image",
        use_grid=cfg.use_grid,
        grid_size=cfg.grid_size,
    )
    ds.transform = ds.get_transforms(cfg.image_size)

    print(
        f"Initiating torch_pipeline on dataset of length {len(ds)} using batch_size {cfg.batch_size} and grid_size {cfg.grid_size}"
    )
    print_current_time()

    embeddings, logits = torch_pipeline(
        ds,
        batch_size=cfg.batch_size,
        use_grid=cfg.use_grid,
        grid_size=cfg.grid_size,
        cpu_count=cfg.cpu_count,
        top_k=cfg.top_k,
        device=cfg.device,
    )

    pred_df = create_predictions_df(ds, embeddings, logits)

    pred_ds = HFDataset.from_pandas(pred_df)
    pred_ds.save_to_disk(cfg.test_embeddings_path)

    print(f"Predictions saved to {cfg.test_embeddings_path}")
    print_current_time()

    print_dir_size(cfg.test_embeddings_dir)


def run_embed_test():
    """

    Create embeddings from the full test set using the DINOv2 model and save them to disk.

    """
    cfg = Config()
    try:
        make_predictions_and_save(cfg)
    except Exception as e:
        print(f"Error in make_predictions_and_save: {e}")

    print("All done!")
    print_current_time()


if __name__ == "__main__":
    print(f"Initiating function run_embed_test() from file: {__file__}")
    print_current_time()

    run_embed_test()
