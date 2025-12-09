from pathlib import Path

import torch
from torch import nn


class FeatureStudyHead(nn.Module):
    """A neural network head for studying features at different levels.

    This class processes input features using pooling and linear layers to produce
    outputs for clean, expansion, oiq, and retro tasks.
    """

    def __init__(self, in_features: int, feature_level: int) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.feature_level = feature_level

        self.clean_head = nn.Sequential(
            nn.Linear(in_features, 3, bias=True),
        )
        self.expansion_head = nn.Sequential(
            nn.Linear(in_features, 3, bias=True),
        )
        self.oiq_head = nn.Sequential(
            nn.Linear(in_features, 3, bias=True),
        )
        self.retro_head = nn.Sequential(
            nn.Linear(in_features, 2, bias=True),
        )

    def forward(
        self, features: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.feature_level == 5:
            pooled = [self.pool(f).flatten(1) for f in features]
            x = torch.cat(pooled, dim=1)
        else:
            x = features[self.feature_level - 1]
            x = self.pool(x).flatten(1)
        return (
            self.clean_head(x),
            self.expansion_head(x),
            self.oiq_head(x),
            self.retro_head(x),
        )


class MlpHead(nn.Module):
    """A neural network head for studying features at different levels.

    This class processes input features using pooling and linear layers to produce
    outputs for clean, expansion, oiq, and retro tasks.
    """

    def __init__(self, in_features: int, weights_path=None) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.clean_head = MLPClassificationHead(
            in_channels=in_features, upsample=4, dropout=0.3, num_classes=3
        )
        self.expansion_head = MLPClassificationHead(
            in_channels=in_features, upsample=4, dropout=0.3, num_classes=3
        )
        self.oiq_head = MLPClassificationHead(
            in_channels=in_features, upsample=4, dropout=0.3, num_classes=3
        )
        self.retro_head = MLPClassificationHead(
            in_channels=in_features, upsample=4, dropout=0.3, num_classes=2
        )

        # Optional weight loading
        if weights_path is not None:
            weights_path = Path(weights_path)
            if not weights_path.is_file():
                raise FileNotFoundError(
                    f"Pretrained weights not found at: {weights_path}"
                )

            state_dict = torch.load(weights_path, map_location="cpu")
            self.load_state_dict(state_dict, strict=True)

    def forward(self, features):
        pooled = [self.pool(f).flatten(1) for f in features]
        x = torch.cat(pooled, dim=1)
        return (
            self.clean_head(x),
            self.expansion_head(x),
            self.oiq_head(x),
            self.retro_head(x),
        )


class MLPClassificationHead(nn.Module):
    """A classification head using MLP layers.

    This class applies a multi-layer perceptron (MLP) to classify input features
    into a specified number of classes.
    """

    def __init__(
        self,
        in_channels: int,
        upsample: int = 4,
        dropout: float = 0.0,
        num_classes: int = 3,
    ) -> None:
        super().__init__()
        hidden_dim = in_channels * upsample
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_dim, bias=True),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes, bias=True),
        )

    def forward(self, x):
        return self.mlp(x)
