from pathlib import Path

import torch
from torch import nn

from models.MetaFormer import caformer_s18


class CaformerBackbone(nn.Module):
    """A backbone model using the CaFormer-S18 architecture for feature extraction.

    This class initializes a CaFormer-S18 model, optionally loads pretrained weights,
    and outputs extracted features.
    """

    def __init__(self, weights_path: str = "", *, freeze: bool = True) -> None:
        super().__init__()

        self.backbone = caformer_s18()
        self.backbone.head = nn.Identity()

        if weights_path:
            weights_path = Path(weights_path)
            if not weights_path.is_file():
                error_message = f"Pretrained weights not found at {weights_path}"
                raise FileNotFoundError(error_message)

            state_dict = torch.load(weights_path, map_location="cpu")
            self.backbone.load_state_dict(state_dict, strict=False)

        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the CaFormer backbone.

        Args:
            img (torch.Tensor): Input image tensor.

        Returns:
            torch.Tensor: A tuple containing:
                - cls (torch.Tensor): Pooled + normalized high-level features (4th feature map) for classification.
                - features (torch.Tensor): Extracted features at multiple spatial resolutions for segmentation:
                  1.(B, 64, 56, 56), 2.(B, 128, 28, 28), 3.(B, 320, 14, 14), 4.(B, 512, 7, 7).
        """
        cls, raw_features = self.backbone(img)

        return cls, raw_features
