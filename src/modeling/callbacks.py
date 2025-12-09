import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback


class PerHeadMetricRecorder(Callback):
    """A PyTorch Lightning callback to record and track the best metrics, logits, and labels.

    For each head during validation epochs.

    Attributes:
        heads (list): List of head names to track.
        best_metrics (dict): Dictionary to store the best metrics for each head.
        best_scores (dict): Dictionary to store the best scores for each head.
        best_logits (dict): Dictionary to store the best logits for each head.
        best_labels (dict): Dictionary to store the best labels for each head.
    """

    def __init__(self, heads: list[str]) -> None:
        super().__init__()
        self.heads = heads
        self.best_metrics = {head: {} for head in heads}
        self.best_scores = {head: float("-inf") for head in heads}
        self.best_logits = {head: None for head in heads}
        self.best_labels = {head: None for head in heads}

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Called at the end of the validation epoch to update the best metrics, logits, and labels."""
        for head in self.heads:
            key = f"val_{head}_macro_auroc"
            current_score = trainer.callback_metrics.get(key)
            if current_score is None:
                continue

            score = float(current_score)
            if score > self.best_scores[head]:
                self.best_scores[head] = score
                self.best_metrics[head] = {
                    k: float(v) for k, v in trainer.callback_metrics.items() if k.startswith(f"val_{head}_")
                }
                self.best_metrics[head][f"{head}_best_epoch"] = trainer.current_epoch

                self.best_logits[head] = torch.cat(pl_module.val_epoch_logits[head], dim=0).detach().cpu()
                self.best_labels[head] = torch.cat(pl_module.val_epoch_labels[head], dim=0).detach().cpu()
