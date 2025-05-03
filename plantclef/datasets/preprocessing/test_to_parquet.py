"""
Created on Friday May 2nd, 2025
Created by Jacob A Rose

Load the directory of test images from plantclef 2025 into a spark dataframe, and write it to a parquet file on disk for later efficient reading during model embedding/inference.

* [TODO] -- Add more details about data locations
* [TODO] -- Add argparse functionality
* [TODO] -- Add S3 integration to minimize local disk costs
"""

import os
from pathlib import Path
from pyspark.sql import DataFrame


def create_spark_test_df(spark, root_dir: Path, output_dir: Path) -> DataFrame:
    """
    Create a parquet file from the test images in the specified directory.

    Args:
        root_dir (Path): The root directory containing the test images.
        output_dir (Path): The output directory for the parquet file.
    """

    image_df = (
        spark.read.format("binaryFile")
        .option("pathGlobFilter", "*.jpg")
        .option("recursiveFileLookup", "true")
        .load(root_dir.as_posix())
    )

    to_remove = "file:" + str(root_dir.parents[0])

    # Remove "file:{base_dir.parents[0]" from path column
    image_df = image_df.withColumn("path", regexp_replace("path", to_remove, ""))

    split_path = split(image_df["path"], "/")
    image_df = image_df.withColumn("file_name", element_at(split_path, -1))

    image_df = image_df.select(
        "path",
        "file_name",
        image_df["content"].alias("data"),
    )

    return image_df


if __name__ == "__main__":
    from plantclef.spark_utils import get_spark
    from pyspark.sql.functions import element_at, regexp_replace, split

    root = "/teamspace/studios/this_studio/plantclef-vision/data"
    test_image_dir = (
        root
        + "plantclef2025/competition-metadata/PlantCLEF2025_test_images/PlantCLEF2025_test_images"
    )
    test_parquet_output_dir = root + "/parquet/plantclef2025/full_test"
    os.makedirs(test_parquet_output_dir, exist_ok=True)

    root_dir = Path(test_image_dir)
    output_dir = Path(test_parquet_output_dir)

    spark = get_spark()

    image_df = create_spark_test_df(spark, root_dir, output_dir)

    image_df.write.mode("overwrite").parquet(str(output_dir))
