import logging
from pathlib import Path

import torch
from models.backbones import CaformerBackbone
from models.heads import FeatureStudyHead, MlpHead
from torch import nn


class ModelBase(nn.Module):
    """A base model class that combines a backbone and a head for classification tasks.

    Args:
        backbone (str): The type of backbone to use.
            Options include 'resnet_pretrained', 'caformer_pretrained', and 'caformer_finetuned'.
        head (str): The type of head to use (e.g., 'mlp_binary', 'mlp_multiclass', 'caformer').
    """

    def __init__(
        self, head: str | None = None, feature_level: int = 4, *, freeze: bool = True
    ) -> None:
        super().__init__()

        current_file_path = Path(__file__).resolve()
        weights_folder = current_file_path.parent.parent.parent / "model_weights"

        self.backbone = CaformerBackbone(
            weights_path=weights_folder / "backbone/ca_s18_finetuned.pth",
            freeze=freeze,
        )

        if feature_level == 1:
            in_features = 64
        elif feature_level == 2:
            in_features = 128
        elif feature_level == 3:
            in_features = 320
        elif feature_level == 4:
            in_features = 512
        elif feature_level == 5:
            in_features = 512 + 320 + 128 + 64
        else:
            error_message = f"Invalid feature level: {feature_level}"
            raise ValueError(error_message)

        if head is None:
            logging.warning("No head is created")
        elif head == "feature_study":
            self.head = FeatureStudyHead(
                in_features=in_features, feature_level=feature_level
            )
        elif head == "mlp":
            self.head = MlpHead(in_features=in_features)
        else:
            error_message = f"Unknown head: {head}"
            raise ValueError(error_message)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output logits from the model.
        """
        cls, raw_features = self.backbone(x)
        logits = self.head(raw_features)
        return logits
