import timm
import torch
import pytorch_lightning as pl

from plantclef.model_setup import setup_fine_tuned_model
from plantclef.config import get_device, get_class_mappings_file


class DINOv2LightningModel(pl.LightningModule):
    """PyTorch Lightning module for extracting embeddings from a fine-tuned DINOv2 model."""

    def __init__(
        self,
        model_path: str = setup_fine_tuned_model(),
        model_name: str = "vit_base_patch14_reg4_dinov2.lvd142m",
        top_k: int = 10,
    ):
        super().__init__()
        self.model_device = get_device()
        self.num_classes = 7806  # total plant species
        self.top_k = top_k

        # load the fine-tuned model
        self.model = timm.create_model(
            model_name,
            pretrained=False,
            num_classes=self.num_classes,
            checkpoint_path=model_path,
            dynamic_img_size=True,
        )

        # load transform
        self.data_config = timm.data.resolve_model_data_config(self.model)
        self.transform = timm.data.create_transform(
            **self.data_config, is_training=False
        )

        # move model to device
        self.model.to(self.model_device)
        self.model.eval()
        # class mappings file for classification
        self.class_mappings_file = get_class_mappings_file()
        # load class mappings
        self.cid_to_spid = self._load_class_mappings()

    def _load_class_mappings(self):
        with open(self.class_mappings_file, "r") as f:
            class_index_to_class_name = {i: line.strip() for i, line in enumerate(f)}
        return class_index_to_class_name

    def forward(self, batch):
        """Extract embeddings using the [CLS] token."""
        with torch.no_grad():
            batch = batch.to(self.model_device)  # move to device

            if batch.dim() == 5:  # (B, grid_size**2, C, H, W)
                B, G, C, H, W = batch.shape
                batch = batch.view(B * G, C, H, W)  # (B * grid_size**2, C, H, W)
            # forward pass
            features = self.model.forward_features(batch)
            embeddings = features[:, 0, :]  # extract [CLS] token
            logits = self.model(batch)

        return embeddings, logits

    def predict_step(self, batch, batch_idx):
        """
        Runs inference on batch and returns embeddings and top-K logits.
        Args:
            batch: Input batch of images.
            batch_idx: Index of the batch.
        Returns:
            embeddings: Extracted embeddings from the model.
            logits: Top-K logits for each image.

        batch can be of either shape [B, C, H, W] or [B, grid_size**2, C, H, W]

        embeddings will be of shape [B, 768] or [B * grid_size**2, 768]
        logits will be List[Dict(species_id (str), probability (float))]
            the list is length of either B or B * grid_size**2
            each dict has k key:value pairs, 1 for each of the top-k logits
        """
        try:
            embeddings, logits = self(batch)
        except Exception as e:
            import pdb

            pdb.set_trace()
            print(f"Error during forward pass: {e}")

        try:
            probabilities = torch.softmax(logits, dim=1)
            top_probs, top_indices = torch.topk(probabilities, k=self.top_k, dim=1)
        except Exception as e:
            import pdb

            pdb.set_trace()
            print(f"Error during softmax/topk: {e}")

        try:
            # map class indices to species names
            batch_logits = []
            for i in range(len(logits)):
                species_probs = {
                    self.cid_to_spid.get(
                        int(top_indices[i, j].item()), "Unknown"
                    ): float(top_probs[i, j].item())
                    for j in range(self.top_k)
                }
                batch_logits.append(species_probs)
        except Exception as e:
            import pdb

            pdb.set_trace()
            print(f"Error during mapping class indices: {e}")

        return embeddings, batch_logits

    def predict_grid_step(self, batch, batch_idx, grid_size=None):
        """
        Runs inference on batch and returns embeddings and top-K logits for each grid tile.
        Args:
            batch: Input batch of images.
            batch_idx: Index of the batch.
            grid_size: Size of the grid (default is None, which infers from the batch shape).
        Returns:
            embeddings: Extracted embeddings from the model.
            logits: Top-K logits for each grid tile.


        batch is expected to be of shape [B * grid_size**2, C, H, W]
        embeddings will be of shape [B, grid_size**2, 768]
        logits will be List[List[Dict(species_id (str), probability (float))]]
            the outer list is length of B
            the inner list is length of grid_size**2
            each dict has k key:value pairs, 1 for each of the top-k logits
        """
        # Extract the grid size from the batch shape
        if grid_size is None:
            grid_size = int(batch.shape[1] ** 0.5)

        embeddings, logits = self.predict_step(batch, batch_idx)

        embeddings = embeddings.view(-1, grid_size**2, 768)
        logits = [
            logits[i : i + grid_size**2] for i in range(0, len(logits), grid_size**2)
        ]

        return embeddings, logits
