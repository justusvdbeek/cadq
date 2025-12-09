import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn
from torchmetrics import Metric


class CE_L1Loss(nn.Module):
    """A loss function combining cross-entropy loss and L1 loss.

    This class computes a weighted combination of cross-entropy loss and L1 loss,
    with an adjustable alpha parameter to balance the two components.

    Args:
        weight (torch.Tensor | None): Optional class weights for the cross-entropy loss.
    """

    def __init__(self, weight: torch.Tensor | None = None) -> None:
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss(weight=weight)
        self.l1_loss = nn.L1Loss(reduction="none")
        self.alpha = 0.2
        self.register_buffer("class_weights", weight if weight is not None else None)
        self.num_classes = 3

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        ce_loss = self.ce_loss(logits, labels)

        probs = F.softmax(logits, dim=-1)
        class_indices = torch.arange(self.num_classes, device=logits.device).float()
        expected = (probs * class_indices).sum(dim=1)

        l1_loss = self.l1_loss(expected, labels.float())

        if self.class_weights is not None:
            sample_weights = self.class_weights[labels]
            l1_loss = l1_loss * sample_weights
        l1_loss = l1_loss.mean()

        scale = ce_loss.detach() / (l1_loss.detach() + 1e-8)
        scaled_l1 = l1_loss * scale

        total_loss = (1 - self.alpha) * ce_loss + self.alpha * scaled_l1
        return total_loss


class MultiHeadCrossEntropyLoss(nn.Module):
    """A loss function that computes the cross-entropy loss for multiple heads.

    This class supports different loss functions for each head, allowing for
    flexibility in handling various types of outputs. By default, it uses
    `CE_L1Loss` for most heads and `CrossEntropyLoss` for the "retro" head.

    Args:
        class_weights (dict[str, torch.Tensor] | None): Optional dictionary
            mapping head names to class weights for the loss computation.
    """

    def __init__(
        self,
        class_weights: dict[str, torch.Tensor] | None = None,
    ) -> None:
        super().__init__()
        self.loss_fns = nn.ModuleDict()
        for head in ["clean", "expansion", "oiq", "retro"]:
            weight = None
            if class_weights and head in class_weights:
                weight = class_weights[head]

            if head == "retro":
                self.loss_fns[head] = nn.CrossEntropyLoss(weight=weight)
            else:
                self.loss_fns[head] = CE_L1Loss(weight=weight)

    def forward(self, logits_dict: dict[str, torch.Tensor], labels_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        losses = [self.loss_fns[head](logits_dict[head], labels_dict[head]) for head in self.loss_fns]
        return sum(losses) / len(losses)

    def get_loss_fn(self, head: str) -> nn.Module:
        return self.loss_fns[head]


class WeightedCrossEntropyMetric(Metric):
    """A metric for computing the weighted cross-entropy loss.

    This class calculates the weighted cross-entropy loss over multiple
    updates and provides the average loss during computation. It supports
    distributed synchronization for multi-GPU training.

    Args:
        weight (torch.Tensor): Class weights for the cross-entropy loss.
        dist_sync_on_step (bool, optional): Synchronize metric state across
            processes at each forward step. Defaults to False.
    """

    def __init__(self, weight: torch.Tensor, *, dist_sync_on_step: bool = False) -> None:
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.register_buffer("weight", weight)

        self.add_state("total_loss", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_samples", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, logits: torch.Tensor, labels: torch.Tensor):
        # Compute weighted cross entropy manually with reduction='sum'
        loss = F.cross_entropy(logits, labels, weight=self.weight, reduction="sum")
        self.total_loss += loss
        self.total_samples += labels.numel()

    def compute(self) -> torch.Tensor:
        if self.total_samples == 0:
            return torch.tensor(0.0, device=self.total_loss.device)
        return self.total_loss / self.total_samples
