"""
file: plantclef-vision/plantclef/utils/df_utils.py
Created on: Saturday May 17th, 2025
Created by Jacob A Rose


utilities for dealing with pandas and polars dataframes


"""

import numpy as np
import pandas as pd
import polars as pl

# def create_partition_column(df, num_partitions):
#     """Creates a partition column based on the number of rows.

#     Args:
#         df: The Pandas DataFrame.
#         num_partitions: The desired number of partitions.

#     Returns:
#         The DataFrame with an added 'partition' column.
#     """
#     df['partition'] = pd.cut(df.index, bins=num_partitions, labels=False, include_lowest=True)
#     return df


def create_partition_column(df, max_rows_per_partition):
    """
    Creates a partition column in a pandas DataFrame based on a maximum number of rows per partition.

    Args:
        df (pd.DataFrame): The input DataFrame.
        max_rows_per_partition (int): The maximum number of rows allowed in each partition.

    Returns:
        pd.DataFrame: The DataFrame with an added 'partition' column.
    """
    num_rows = len(df)
    num_partitions = int(np.ceil(num_rows / max_rows_per_partition))

    # df['partition'] = np.repeat(range(num_partitions), max_rows_per_partition)[:num_rows]
    df = df.assign(
        partition=np.repeat(range(num_partitions), max_rows_per_partition)[:num_rows]
    )
    return df


def save_df_to_parquet(
    df: pd.DataFrame,  # type: ignore
    path: str,
    max_rows_per_partition: int = 10000,
):
    """
    Saves a pandas DataFrame to a parquet file with partitioning.
    Args:
        df (pd.DataFrame): The DataFrame to save.
        path (str): The path where the parquet file will be saved.
        max_rows_per_partition (int): The maximum number of rows per partition.
    """
    if not path.endswith(".parquet"):
        path += ".parquet"
    df = create_partition_column(df, max_rows_per_partition=max_rows_per_partition)  # type: ignore
    df: pl.DataFrame = pl.from_pandas(df)  # type: ignore
    df.write_parquet(path, partition_by="partition")
    return path
