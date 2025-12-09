import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
from torchmetrics.classification import (
    MulticlassAUROC,
    MulticlassAveragePrecision,
)

from metrics import MacroMAE
from modeling.losses import MultiHeadCrossEntropyLoss, WeightedCrossEntropyMetric


class ClassificationModule(pl.LightningModule):
    """A PyTorch Lightning module for binary and multiclass classification tasks.

    This module wraps a given model and provides functionality for training, validation,
    and testing, including metrics computation and logging. It supports binary and
    multiclass classification with configurable loss functions, metrics, and learning rate.
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float,
        pos_weights: dict[str, torch.Tensor] | None = None,
        epochs: int = 20,
        scheduler: str = "cosine",
    ) -> None:
        super().__init__()

        self.model = model
        self.criterion = MultiHeadCrossEntropyLoss(class_weights=pos_weights)
        self.scheduler = scheduler
        self.lr = float(torch.tensor(lr, dtype=torch.float32))
        self.num_classes_per_head = {"clean": 3, "expansion": 3, "oiq": 3, "retro": 2}
        self.heads = ["oiq", "expansion", "clean", "retro"]
        self.max_epochs = epochs
        self.pos_weights = pos_weights or {head: None for head in self.heads}
        self.head_epochs = {"clean": 20, "expansion": 20, "oiq": 20, "retro": 20}

        self.save_hyperparameters({"lr": lr, "num_classes_per_head": self.num_classes_per_head})
        self.metrics = self._init_metrics()
        self.val_epoch_logits = {head: [] for head in self.heads}
        self.val_epoch_labels = {head: [] for head in self.heads}

    def _init_metrics(self) -> nn.ModuleDict:
        def get_metric_dict(stage: str, head: str) -> dict[str, nn.Module]:
            metrics = {}
            num_classes = self.num_classes_per_head[head]
            if stage in {"val", "test"}:
                metrics.update(
                    {
                        "weighted_ce": WeightedCrossEntropyMetric(weight=self.pos_weights[head]),
                        "macro_mae": MacroMAE(num_classes=num_classes),
                        "macro_auroc": MulticlassAUROC(num_classes=num_classes, average="macro"),
                        "macro_auprc": MulticlassAveragePrecision(num_classes=num_classes, average="macro"),
                        "per_class_auprc": MulticlassAveragePrecision(num_classes=num_classes, average=None),
                    }
                )

            return metrics

        stages = ["train", "val", "test"]

        return nn.ModuleDict(
            {
                f"{stage}_metrics": nn.ModuleDict(
                    {head: nn.ModuleDict(get_metric_dict(stage, head)) for head in self.heads}
                )
                for stage in stages
            }
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def configure_optimizers(self) -> dict:
        optimizer_groups = []

        head_lrs = {
            "clean": self.lr,
            "expansion": self.lr,
            "oiq": self.lr,
            "retro": self.lr,
        }

        for head in self.heads:
            head_module = getattr(self.model.head, f"{head}_head", None)
            if head_module is None:
                error_message = f"Head {head} not found in model"
                raise ValueError(error_message)

            decay, no_decay = [], []

            for name, param in head_module.named_parameters():
                if not param.requires_grad:
                    continue
                if name.endswith(".bias") or "norm" in name.lower():
                    no_decay.append(param)
                else:
                    decay.append(param)

            head_lr = head_lrs.get(head, self.lr)

            optimizer_groups.append({"params": decay, "weight_decay": 1e-2, "lr": head_lr, "name": head + "_decay"})
            optimizer_groups.append(
                {"params": no_decay, "weight_decay": 0.0, "lr": head_lr, "name": head + "_no_decay"}
            )

        decay, no_decay = [], []
        for name, param in self.model.backbone.named_parameters():
            if not param.requires_grad:  # skip if frozen
                continue
            if name.endswith(".bias") or "norm" in name.lower():
                no_decay.append(param)
            else:
                decay.append(param)

        if decay or no_decay:  # only add if unfrozen
            backbone_lr = 1e-5
            optimizer_groups.append(
                {"params": decay, "weight_decay": 1e-4, "lr": backbone_lr, "name": "backbone_decay"}
            )
            optimizer_groups.append(
                {"params": no_decay, "weight_decay": 0.0, "lr": backbone_lr, "name": "backbone_no_decay"}
            )

        optimizer = torch.optim.AdamW(optimizer_groups)

        if self.scheduler == "cosine":
            # Use the *maximum* T_max across heads as a conservative schedule
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.max_epochs,
                eta_min=1e-6,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                    "frequency": 1,
                },
            }

        if self.scheduler == "none":
            return {"optimizer": optimizer}

        error_message = f"Unknown scheduler: {self.scheduler}. Supported: 'cosine', 'none'."
        raise ValueError(error_message)

    def step(self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], stage: str):
        ignore_index = -100

        images, y_clean, y_expansion, y_oiq, y_retro = batch
        labels_dict = {
            "clean": y_clean,
            "expansion": y_expansion,
            "oiq": y_oiq,
            "retro": y_retro,
        }
        masks = {k: v != ignore_index for k, v in labels_dict.items()}

        logits_clean, logits_expansion, logits_oiq, logits_retro = self(images)
        logits_dict = {
            "clean": logits_clean,
            "expansion": logits_expansion,
            "oiq": logits_oiq,
            "retro": logits_retro,
        }

        logits_dict_masked = {k: v[masks[k]] for k, v in logits_dict.items()}
        labels_dict_masked = {k: v[masks[k]] for k, v in labels_dict.items()}

        head_losses = {}

        current_epoch = self.current_epoch

        for head in self.heads:
            if current_epoch >= self.head_epochs[head]:
                continue

            if masks[head].sum() == 0:
                continue

            logits = logits_dict_masked[head]
            y = labels_dict_masked[head]

            head_loss = self.criterion.loss_fns[head](logits, y)
            head_losses[head] = head_loss

            if stage in {"train", "val"}:
                self.log(
                    f"{stage}_{head}_loss",
                    head_loss,
                    on_epoch=True,
                    on_step=False,
                )

            if stage in {"val", "test"}:
                self.val_epoch_logits[head].append(logits.detach())
                self.val_epoch_labels[head].append(y.detach())

        total_loss = sum(head_losses.values()) / len(head_losses)
        if stage == "train":
            self.log(
                f"{stage}_loss",
                total_loss,
                on_epoch=True,
                prog_bar=True,
                on_step=False,
            )

        return total_loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self.step(batch, "val")

    def test_step(self, batch, batch_idx):
        self.step(batch, "test")

    def _log_epoch_metrics(self, stage: str):
        self.model.eval()
        for head in self.heads:
            logits = torch.cat(self.val_epoch_logits[head], dim=0)
            y = torch.cat(self.val_epoch_labels[head], dim=0)
            num_classes = self.num_classes_per_head[head]
            probs = F.softmax(logits, dim=1)
            expected_value = (
                (probs * torch.arange(num_classes, device=probs.device)).sum(dim=1) if num_classes > 2 else probs[:, 1]
            )

            metric_group = self.metrics[f"{stage}_metrics"][head]

            for name, metric in metric_group.items():
                if name == "weighted_ce":
                    metric.update(logits, y)
                elif name == "macro_mae":
                    metric.update(expected_value, y)
                elif name in {"macro_auroc", "macro_auprc", "per_class_auprc"}:
                    metric.update(probs, y)

            for name, metric in metric_group.items():
                try:
                    val = metric.compute()
                except (RuntimeError, ValueError):
                    continue

                if isinstance(val, torch.Tensor) and val.ndim == 1:
                    for i, v in enumerate(val):
                        self.log(
                            f"{stage}_{head}_{name}_{i}",
                            v,
                            on_step=False,
                            on_epoch=True,
                        )
                else:
                    self.log(
                        f"{stage}_{head}_{name}",
                        val,
                        on_step=False,
                        on_epoch=True,
                    )

        self.val_epoch_logits = {head: [] for head in self.heads}
        self.val_epoch_labels = {head: [] for head in self.heads}

    def _reset_metrics(self, stage: str):
        for head in self.heads:
            metric_group = self.metrics[f"{stage}_metrics"][head]
            for metric in metric_group.values():
                metric.reset()

    def on_validation_epoch_end(self):
        self._log_epoch_metrics(stage="val")
        self._reset_metrics(stage="val")

    def on_test_epoch_end(self):
        self._log_epoch_metrics(stage="test")
        self._reset_metrics(stage="test")
