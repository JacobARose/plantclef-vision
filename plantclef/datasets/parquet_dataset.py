"""
Created on Friday May 2nd, 2025
Created by Jacob A Rose



"""

from torch.utils.data import Dataset
import pyarrow.parquet as pq
import io
from PIL import Image


class ParquetImageDataset(Dataset):
    def __init__(self, parquet_file, image_col="image", transform=None):
        self.parquet_file = parquet_file
        self.image_col = image_col
        self.transform = transform
        self.table = pq.read_table(parquet_file)
        self.image_data = self.table.column(self.image_col).to_numpy()
        self.length = len(self.image_data)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        image_bytes = self.image_data[idx]
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image
