import torch
import faiss
import pandas as pd
import polars as pl
from plantclef.config import get_device


class FaissClassifier:
    def __init__(self, train_df: pd.DataFrame | pl.DataFrame):
        """
        :param train_df: DataFrame with columns ["species_id", "embeddings"]
        """
        self.device = get_device()
        self.index, self.idx2cls = self.build_index(train_df)

    def build_index(self, train_df):
        """Builds the FAISS index from the training data."""

        # store class labels
        # idx2cls = train_df["image_name"].values
        # # convert embeddings to tensor
        # embs_array = np.array(train_df["embeddings"].tolist(), dtype=np.float32)
        # embs = torch.tensor(embs_array, device=self.device)
        # # normalize embeddings for cosine similarity
        # embs = torch.nn.functional.normalize(embs, p=2, dim=1)

        idx2cls = train_df["image_name"].to_numpy()

        embed_array = (
            train_df.drop("image_name", "partition").cast(pl.Float32).to_numpy()
        )
        embs = torch.tensor(embed_array)
        embs = torch.nn.functional.normalize(embs, p=2, dim=1)

        # create FAISS index
        index = faiss.IndexFlatIP(embs.shape[1])  # inner product (dot product)
        index.add(embs.cpu().numpy())  # FAISS expects numpy arrays # type: ignore

        return index, idx2cls

    def make_prediction(self, query_embeddings: torch.Tensor, k=1):
        """
        Predicts the class of given embeddings using nearest neighbor search.
        :param query_embeddings: tensor of shape (N, D) where N is the number of embeddings and D is the embedding dimension
        :param k: number of nearest neighbors to return
        :return: predictions, similarities
        """

        # normalize embeddings for cosine similarity
        query_embeddings = torch.nn.functional.normalize(query_embeddings, p=2, dim=1)
        # perform search
        similarities, indices = self.index.search(query_embeddings.cpu().numpy(), k=k)
        predictions = self.idx2cls[indices]
        return predictions, similarities
