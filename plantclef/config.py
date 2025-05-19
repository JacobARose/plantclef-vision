import os
import csv
import torch
import pandas as pd
from pathlib import Path
from collections import Counter

import json
from dataclasses import dataclass, asdict
from rich.repr import auto
from rich import print as pprint
from typing import Optional, Dict, Any

# T = TypeVar("T")


@dataclass
@auto
class BaseConfig:
    config_path: str = "config.json"

    def to_dict(self):
        """Convert the dataclass to a dictionary."""
        out = asdict(self)
        for k in list(out.keys()):
            if out[k] is None:
                del out[k]
        return out

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        """Create an instance of the dataclass from a dictionary."""
        return cls(**data)

    def save(
        self, path: str, indent: Optional[int] = None, exist_ok: bool = True
    ) -> str:
        """Save the dataclass to a JSON file.

        To load, use `cfg = Config.load(path)`.

        Args:
            filename: Path to save the file. Should end with .json
            indent: Optional indentation for pretty printing (None for compact)
        """
        path = path or self.config_path
        if not path.endswith(".json"):
            path += ".json"
        path = os.path.abspath(os.path.expanduser(path))
        os.makedirs(os.path.dirname(path), exist_ok=exist_ok)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=indent)
        return path

    @classmethod
    def load(cls, filename: str):
        """Load the dataclass from a JSON file.

        Args:
            filename: Path to the JSON file

        Returns:
            An instance of the dataclass
        """
        with open(filename, "r") as f:
            data = json.load(f)
        return cls(**data)

    def __rich_repr__(self):
        # Get all fields from the dataclass
        for field_name in self.__dataclass_fields__:
            value = getattr(self, field_name)
            # Always show the field (no default hiding)
            yield field_name, value, None

    def show(self):
        """Print the dataclass in a readable format."""
        pprint(self)


def get_torch_version():
    return torch.__version__


def get_device():
    """Get the device (CPU, GPU, or XPU) available for PyTorch."""
    if torch.xpu.is_available():
        device = "xpu"
    elif torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    return device


def get_base_dir() -> str:
    """Get the base project directory."""
    return Path(__file__).resolve().parent.parent


def get_class_mappings_file() -> str:
    """Get the directory containing the class mappings for the DINOv2 model."""
    base_dir = get_base_dir()
    return f"{base_dir}/plantclef/class_mapping.txt"


def parse_integer_list(string):
    """
    Parses a string containing a list of integers and returns the list of integers.

    :param string: A string in the format "[1, 2, 3]"
    :return: A list of integers [1, 2, 3]
    """
    # Remove brackets and split by commas
    string = string.strip("[]")
    if not string:
        return []
    return [int(x.strip()) for x in string.split(",")]


def create_submission_csv(
    faiss_df: pd.DataFrame,
    output_path: str = "../data/submission/submission.csv",
    species_col: str = "pred_species_ids",
    save_csv: bool = False,
    limit_top_k: Optional[int] = None,
):
    """
    Aggregates FAISS-predicted species IDs across 3x3 tiles and writes a submission CSV file.

    :param faiss_df: DataFrame with FAISS predictions, including a column `species_ids` or `pred_species_ids`
    :param output_path: Path to write the CSV file
    :param species_col: Column containing predicted species IDs for each tile
    """

    # group by image_name (each quadrat/image), collect all tile-level species IDs
    grouped = faiss_df.groupby("image_name")[species_col].apply(list)
    records = []
    for image_name, species_id_lists in grouped.items():
        # flatten species_id lists from all tiles
        flat_ids = [
            sid for sublist in species_id_lists for sid in sublist if sid is not None
        ]
        # count and sort by frequency (optional for top species prioritization)
        counted = Counter(flat_ids)
        sorted_ids = [species_id for species_id, _ in counted.most_common()]
        # deduplicate + keep sorted by frequency
        unique_sorted_ids = list(dict.fromkeys(sorted_ids))
        # format as double-bracketed string
        species_ids_str = f"[{', '.join(str(sid) for sid in unique_sorted_ids)}]"
        records.append({"quadrat_id": image_name, "species_ids": species_ids_str})

    # build final DataFrame and write to CSV
    df_run = pd.DataFrame(records)

    if limit_top_k is not None:
        df_run = df_run.apply(
            lambda x: parse_integer_list(x["species_ids"])[:limit_top_k], axis=1
        )

    if save_csv:
        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)
        df_run.to_csv(output_path, sep=",", index=False, quoting=csv.QUOTE_ALL)
        print(f"Submission file saved to: {output_path}")
    return df_run


if __name__ == "__main__":
    # get root directory
    base_dir = get_base_dir()
    print("Base directory:", base_dir)

    # get class mappings file
    class_mappings_file = get_class_mappings_file()
    print("Class mappings file:", class_mappings_file)
