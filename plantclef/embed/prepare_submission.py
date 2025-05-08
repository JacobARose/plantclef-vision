"""
file: prepare_submission.py
Created on: Wednesday May 7th, 2025
Created by: Jacob A Rose

Description:
    This script prepares the submission files for the PlantCLEF 2025 competition.
    It loads the test logits and creates a submission file in the required format.
    The submission file is saved to the specified directory.


prerequisites:
    - [TODO] - Move the image directory -> HFDataset.save_to_disk() code from `embedding.ipynb` to a dedicated script.
    - Run the `plantclef/embed/workflow.py` script to generate the logits and embeddings.

"""

import csv
from datasets import Dataset as HFDataset
from more_itertools import flatten
import more_itertools as mit
import pandas as pd
from typing import Tuple, List
from plantclef.embed.workflow import Config
from plantclef.embed.utils import print_dir_size, print_current_time


def remove_NaN_values_from_dict(d: dict) -> dict:
    """Remove NaN values from a dictionary."""
    return {k: v for k, v in d.items() if v is not None}


def sort_and_filter_dict(d: dict, top_k: int = 0) -> List[Tuple[str, float]]:
    """
    Sort a dictionary by values and filter out NaN values.

    Takes in a dictionary mapping str keys to float values, then
        removes any keys with NaN values
        sorts the remaining key-value pairs in descending order by value and transforms into a sorted list of tuples.
    If top_k is specified, only the top_k items are returned.
    Args:
        d (dict): The dictionary to sort and filter.
        top_k (int): The number of top items to return. Default is 0, which returns all items.
    Returns:
        List[Tuple[str, float]]: A sorted list of tuples (key, value) from the dictionary.

    * [TODO] -- Consider adding a threshold for the values to filter out low-confidence predictions.

    """
    # Remove NaN values
    d = remove_NaN_values_from_dict(d)

    # Sort the dictionary by values in descending order
    sorted_list = sorted(d.items(), key=lambda item: item[1], reverse=True)

    # If top_k is specified, return only the top_k items
    if top_k > 0:
        sorted_list = sorted_list[:top_k]

    return sorted_list


def select_top_k_unique_logits(df: pd.DataFrame, top_k: int = 5) -> list:
    """
    Select the top k unique logits from the DataFrame.
    """
    assert df.shape == (9, 2)

    # img_name, g = next(iter(dfg))
    logits = sorted(flatten(df["logits"].to_list()), key=lambda x: x[1], reverse=True)
    logits_unique = list(mit.unique_everseen(logits, key=lambda x: x[0]))

    top_k_logits_unique = logits_unique[:top_k]

    return top_k_logits_unique


def groupby_image_select_top_k_unique_logits(
    df: pd.DataFrame, top_k: int = 5
) -> pd.DataFrame:
    """
    Group by image across all image tiles and select the top k unique logits.

    Args:
        df (pd.DataFrame): The DataFrame to process.
        top_k (int): The number of top items to include. Default is 5.
    Returns:
        pd.DataFrame: A DataFrame containing the selected logits.
    """

    if isinstance(df, pd.DataFrame):
        df = df.groupby("image_name")

    return (
        df.apply(select_top_k_unique_logits, top_k=top_k).rename("logits").reset_index()
    )


def load_test_logits(path: str) -> HFDataset:
    """Load the test logits HuggingFace dataset from the specified path.
    Args:
        path (str): Path to the test logits file.
    Returns:
        HFDataset: The loaded test logits dataset.
    """

    ds = HFDataset.load_from_disk(path)

    # species_ids = [
    #     int(species_id) for species_id in sorted(list(set(ds[0]["logits"].keys())))
    # ]
    # print(f"len(species_ids): {len(species_ids)}")

    return ds


def prepare_submission(ds: HFDataset, top_k: int = 5) -> pd.DataFrame:
    """
    Prepare a submission CSV file from logits saved to disk as a Hugging Face dataset.

    Args:
        ds (HFDataset): The dataset to process.
            Expected columns are  ['image_name', 'embeddings', 'logits', 'tile'].
        top_k (int): The number of top items to include in the submission. Default is 5.

    Returns:
        df (pd.DataFrame): A DataFrame containing the formatted top_k species_id predictions.
            Expected columns are ['quadrat_id', 'species_ids'].


    """
    # Convert the ["image_name", "logits"] columns from the hf dataset in to a pd.DataFrame
    df = ds.remove_columns(["embeddings", "tile"]).to_pandas()

    df = df.assign(
        logits=df.apply(
            lambda x: sort_and_filter_dict(  # Sort species IDs from high-to-low confidence scores remove all but the top_k
                x["logits"], top_k=top_k
            ),
            axis=1,
        )
    )

    df = groupby_image_select_top_k_unique_logits(df, top_k=5)
    df = df.rename(
        columns={
            "image_name": "quadrat_id",
            "logits": "species_ids",
        }
    )
    df = df.assign(
        species_ids=df.apply(
            lambda x: [  # Select only the species IDs
                species_id for species_id, _ in x["species_ids"]
            ],
            axis=1,
        )
    )

    return df


def main():
    cfg = Config()

    # Load the test logits dataset
    ds = load_test_logits(cfg.test_embeddings_path)

    # Prepare the submission DataFrame
    df = prepare_submission(ds, top_k=5)

    # Save the submission DataFrame to a CSV file

    df.to_csv(cfg.test_submission_path, sep=",", index=False, quoting=csv.QUOTE_ALL)

    print("Submission file created successfully!")
    print(f"Submission file saved to {cfg.test_submission_path}")
    print_dir_size(cfg.test_submission_path)

    cfg.save(path=cfg.config_path, indent=4)
    print(f"Config file saved to {cfg.config_path}")
    cfg.show()


if __name__ == "__main__":
    print(f"[INFO] -- script {__file__} started at {print_current_time()}")

    main()

    print(
        f"[SUCCESS] -- script {__file__} completed successfully at {print_current_time()}"
    )
