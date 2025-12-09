import torch
from torch.nn import functional
from torchmetrics import Metric


class MacroCE(Metric):
    """A PyTorch Metric class for computing the macro-averaged cross-entropy (CE) loss."""

    def __init__(self, num_classes: int = 3, *, dist_sync_on_step: bool = False) -> None:
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.num_classes = num_classes
        self.add_state("total_ce", default=torch.zeros(num_classes), persistent=True)
        self.add_state("class_counts", default=torch.zeros(num_classes), persistent=True)

    def update(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        for c in range(self.num_classes):
            mask = targets == c
            num_c = mask.sum()
            if num_c == 0:
                continue

            logits_c = logits[mask]
            targets_c = targets[mask]
            ce_c = functional.cross_entropy(logits_c, targets_c, reduction="mean")

            self.total_ce[c] += ce_c.detach() * num_c
            self.class_counts[c] += num_c

    def compute(self):
        ce_per_class = torch.zeros_like(self.total_ce)
        valid_mask = self.class_counts > 0
        ce_per_class[valid_mask] = self.total_ce[valid_mask] / self.class_counts[valid_mask]
        mean_ce = ce_per_class[valid_mask].mean() if valid_mask.any() else torch.tensor(float("nan"))
        return mean_ce


class MacroMAE(Metric):
    """A PyTorch Metric class for computing the macro-averaged mean absolute error (MAE)."""

    def __init__(self, num_classes: int = 3, *, dist_sync_on_step: bool = False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.num_classes = num_classes
        self.add_state("total_mae", default=torch.zeros(num_classes), persistent=True)
        self.add_state("class_counts", default=torch.zeros(num_classes), persistent=True)

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        for c in range(self.num_classes):
            mask = targets == c
            if mask.sum() == 0:
                continue
            mae_c = (preds[mask] - targets[mask].float()).abs().mean()
            self.total_mae[c] += mae_c * mask.sum()
            self.class_counts[c] += mask.sum()

    def compute(self):
        valid = self.class_counts > 0
        mae_per_class = torch.zeros_like(self.total_mae)
        mae_per_class[valid] = self.total_mae[valid] / self.class_counts[valid]
        return mae_per_class[valid].mean()
