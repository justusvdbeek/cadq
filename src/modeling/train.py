import os
import random
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import wandb
from config import get_args
from data_module import ImageDataModule
from dataset import load_image_dataframe
from modeling.callbacks import PerHeadMetricRecorder
from models.model_base import ModelBase
from models.model_module import ClassificationModule
from pandas import DataFrame
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from scipy import optimize
from sklearn.metrics import balanced_accuracy_score, recall_score
from sklearn.model_selection import GroupKFold


def seed_everything(seed: int) -> None:
    """Set the random seed for reproducibility across various libraries."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    pl.seed_everything(seed, workers=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_class_weight(labels: torch.Tensor, num_classes: int = 3) -> torch.Tensor:
    """Compute class weights from a tensor based on , safely ignoring NaNs."""
    labels = labels.view(-1)

    if labels.is_floating_point():
        labels = labels[~torch.isnan(labels)]

    if labels.numel() == 0:
        return torch.ones(num_classes) / num_classes

    labels = labels.to(torch.long)
    counts = torch.bincount(labels, minlength=num_classes).float()
    weights = counts.sum() / (counts + 1e-6)
    return weights / weights.sum()


def compute_class_weights(df: pd.DataFrame) -> dict:
    """Compute class weights for multiple classification heads."""
    return {
        "clean": compute_class_weight(
            torch.tensor(df["mucosal_cleaning"].values), num_classes=3
        ),
        "expansion": compute_class_weight(
            torch.tensor(df["expansion"].values), num_classes=3
        ),
        "retro": compute_class_weight(
            torch.tensor(df["retrograde"].values), num_classes=2
        ),
        "oiq": compute_class_weight(torch.tensor(df["oiq"].values), num_classes=3),
    }


def main() -> None:
    """Main function to set up and execute the training process."""
    args = get_args()
    seed = args.seed
    seed_everything(seed)

    run_name = f"{args.tag}"
    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df, test_df = load_dataframes(args.data_path)

    if args.final_training:
        train_final_model(args, train_df, test_df, run_name, output_dir)

    else:
        all_fold_scores = perform_cross_validation(
            args, train_df, run_name, output_dir, seed
        )
        agg_name = f"Results_{run_name}"
        run = wandb.init(project=args.wandb_project, name=agg_name, group=agg_name)
        wandb_metrics = summarize_cross_validation(all_fold_scores)
        run.summary.update(wandb_metrics)
        run.finish()


def load_dataframes(data_path: str) -> tuple:
    """Load training and testing dataframes."""
    train_df = load_image_dataframe(data_path, split="train")
    test_df = load_image_dataframe(data_path, split="test")
    return train_df, test_df


def perform_cross_validation(
    args: Namespace, train_df: DataFrame, run_name: str, output_dir: Path, seed: int
) -> list:
    """Perform cross-validation and return fold scores."""
    group_kfold = GroupKFold(n_splits=5, shuffle=True, random_state=seed)
    groups = train_df["patient_id"]
    all_fold_scores = []

    for fold, (train_idx, val_idx) in enumerate(
        group_kfold.split(train_df, groups=groups)
    ):
        fold_scores = train_and_validate_fold(
            args, train_df, train_idx, val_idx, run_name, output_dir, fold
        )
        all_fold_scores.append(fold_scores)

    return all_fold_scores


def train_and_validate_fold(
    args: Namespace,
    train_df: DataFrame,
    train_idx: list,
    val_idx: list,
    run_name: str,
    output_dir: Path,
    fold: int,
) -> dict:
    """Train and validate a single fold."""
    run_name_fold = f"{run_name}_fold{fold+1}"
    wandb_logger = WandbLogger(
        project=args.wandb_project, name=run_name_fold, group=args.tag
    )
    output_dir_fold = output_dir / f"fold_{fold+1}"
    output_dir_fold.mkdir(parents=True, exist_ok=True)

    train_df_fold = train_df.iloc[train_idx]
    val_df_fold = train_df.iloc[val_idx]

    pos_weights = compute_class_weights(train_df_fold)
    data_module = create_data_module(train_df_fold, val_df_fold, args)
    model = create_model(args, pos_weights)

    heads = ["oiq", "expansion", "clean", "retro"]
    recorder = PerHeadMetricRecorder(heads=heads)
    checkpoints = [
        ModelCheckpoint(
            dirpath=output_dir_fold / head,
            filename=f"{run_name_fold}_{head}_epoch={{epoch:02d}}-val_macro_auroc={{val_{head}_macro_auroc:.2f}}",
            monitor=f"val_{head}_macro_auroc",
            mode="max",
            save_top_k=1,
            save_last=False,
        )
        for head in heads
    ]

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        log_every_n_steps=10,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        logger=wandb_logger,
        callbacks=[recorder, *checkpoints],
    )

    trainer.fit(model, datamodule=data_module)
    wandb_logger.experiment.finish()

    metrics = {"fold": fold + 1}
    for head in heads:
        head_metrics = recorder.best_metrics.get(head, {})
        metrics.update(head_metrics)

        logits = recorder.best_logits.get(head)
        labels = recorder.best_labels.get(head)
        if logits is None or labels is None:
            continue

        try:
            result = optimize_thresholds(logits, labels)
        except (ValueError, RuntimeError) as e:
            print(f"Error optimizing thresholds for {head}: {e}")
            continue

        # store thresholds
        if "t1" in result and "t2" in result:
            metrics[f"val_{head}_t1"] = result["t1"]
            metrics[f"val_{head}_t2"] = result["t2"]
        elif "threshold" in result:
            metrics[f"val_{head}_t1"] = result["threshold"]

        # store balanced accuracy and class-wise recalls
        if "balanced_accuracy" in result:
            metrics[f"val_{head}_balanced_accuracy"] = result["balanced_accuracy"]
        if "tpr" in result:
            metrics[f"val_{head}_tpr"] = result["tpr"]
        if "tnr" in result:
            metrics[f"val_{head}_tnr"] = result["tnr"]

    return metrics


def create_data_module(
    train_df_fold: DataFrame, val_df_fold: DataFrame, args: Namespace
) -> ImageDataModule:
    """Create the data module for training and validation."""
    return ImageDataModule(
        dataframes={"train": train_df_fold, "val": val_df_fold, "test": None},
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )


def create_model(args: Namespace, pos_weights: dict) -> ClassificationModule:
    """Create the model for training."""
    model = ModelBase(
        head=args.head,
        feature_level=args.feature_level,
        freeze=args.freeze,
    )
    return ClassificationModule(
        model=model,
        lr=args.lr,
        pos_weights=pos_weights,
        epochs=args.epochs,
        scheduler=args.scheduler,
    )


def optimize_thresholds(logits: torch.Tensor, labels: torch.Tensor) -> dict:
    """Optimize thresholds for classification based on logits and labels.

    Args:
        logits (torch.Tensor): The raw output logits from the model.
        labels (torch.Tensor): The ground truth labels.

    Returns:
        dict: A dictionary containing the optimized thresholds and F1 scores.
    """
    num_classes = logits.shape[1]
    probs = torch.softmax(logits, dim=-1)

    if num_classes == 2:
        class1_probs = probs[:, 1]

        def balanced_accuracy_loss(thresh: float) -> float:
            preds = (class1_probs >= thresh).int()
            return -balanced_accuracy_score(labels.cpu(), preds.cpu())

        result = optimize.minimize_scalar(
            balanced_accuracy_loss, bounds=(0.05, 0.95), method="bounded"
        )
        best_t = result.x
        preds = (class1_probs >= best_t).int()

        ba = balanced_accuracy_score(labels.cpu(), preds.cpu())
        tpr = recall_score(
            labels.cpu(), preds.cpu(), pos_label=1, zero_division=0
        )  # OOD recall
        tnr = recall_score(
            labels.cpu(), preds.cpu(), pos_label=0, zero_division=0
        )  # ID recall
        return {
            "threshold": float(best_t),
            "balanced_accuracy": float(ba),
            "tpr": float(tpr),
            "tnr": float(tnr),
        }

    if num_classes == 3:
        expected_values = (probs * torch.arange(3, device=probs.device).float()).sum(
            dim=1
        )
        expected_values_np = expected_values.cpu().numpy()
        labels_np = labels.cpu().numpy()

        best_ba = -1.0
        best_t1, best_t2 = 0.5, 1.5

        # Grid search: t1 in [0.05, 1.8], t2 in [t1+0.01, 1.99]
        for t1 in np.arange(0.05, 1.80, 0.01):
            for t2 in np.arange(t1 + 0.01, 1.99, 0.01):
                preds = np.digitize(expected_values_np, [t1, t2])
                ba = balanced_accuracy_score(labels_np, preds)
                if ba > best_ba:
                    best_ba = ba
                    best_t1, best_t2 = t1, t2

        # Final predictions with best thresholds
        preds = np.digitize(expected_values_np, [best_t1, best_t2])
        ba = balanced_accuracy_score(labels_np, preds)

        return {
            "t1": float(best_t1),
            "t2": float(best_t2),
            "balanced_accuracy": float(ba),
        }

    raise ValueError(f"Unsupported number of classes: {num_classes}. Expected 2 or 3.")


def summarize_cross_validation(all_fold_scores) -> dict:
    """Summarize cross-validation results and log metrics.

    Args:
        all_fold_scores (list): A list of dictionaries containing metrics for each fold.

    Returns:
        dict: A dictionary containing aggregated metrics across folds.
    """
    print("\n========== Cross Validation Summary ==========\n")

    heads = ["oiq", "expansion", "clean", "retro"]
    metrics_to_keep = [
        "weighted_ce",
        "macro_mae",
        "macro_auroc",
        "per_class_auprc_0",
        "per_class_auprc_1",
        "per_class_auprc_2",
        "best_epoch",
        "balanced_accuracy",
        "tpr",
        "tnr",
        "t1",
        "t2",
    ]

    df = pd.DataFrame(all_fold_scores)
    wandb_metrics = {}
    rows = []
    for head in heads:
        row = [head]
        for metric in metrics_to_keep:
            key = (
                f"val_{head}_{metric}"
                if metric != "best_epoch"
                else f"{head}_best_epoch"
            )
            if key not in df.columns:
                row.append("N/A")
                continue

            values = df[key].astype(float).values
            mean = values.mean()
            std = values.std()

            formatted = f"{mean:.3f} ± {std:.3f}"
            row.append(formatted)

            wandb_metrics[f"cv_mean_{head}_{metric}"] = float(mean)
            wandb_metrics[f"cv_std_{head}_{metric}"] = float(std)
        rows.append(row)

    # Create table
    headers = ["Head"] + metrics_to_keep
    col_widths = [
        max(len(str(row[i])) for row in [headers] + rows) for i in range(len(headers))
    ]

    def format_row(row: list) -> str:
        return " | ".join(f"{str(cell):<{w}}" for cell, w in zip(row, col_widths))

    print(format_row(headers))
    print("-+-".join("-" * w for w in col_widths))

    for row in rows:
        print(format_row(row))

    return wandb_metrics


def train_final_model(
    args: Namespace,
    train_df: DataFrame,
    test_df: DataFrame,
    run_name: str,
    output_dir: Path,
) -> None:
    """Train the final model on the full dataset."""
    pos_weights = compute_class_weights(train_df)
    data_module_full = ImageDataModule(
        dataframes={"train": train_df, "val": None, "test": test_df},
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    final_model = create_model(args, pos_weights)

    final_output_dir = output_dir / "final_model"
    final_output_dir.mkdir(parents=True, exist_ok=True)

    final_checkpoint = ModelCheckpoint(
        dirpath=final_output_dir,
        filename="final_model_{epoch:02d}-{train_loss:.2f}",
        monitor="train_loss",
        mode="min",
        save_top_k=1,
        save_last=False,
    )

    final_logger = WandbLogger(project=args.wandb_project, name=f"{run_name}_final")

    final_trainer = pl.Trainer(
        max_epochs=args.epochs,
        log_every_n_steps=10,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        logger=final_logger,
        callbacks=[final_checkpoint],
        num_sanity_val_steps=0,
    )

    final_trainer.fit(final_model, datamodule=data_module_full)


if __name__ == "__main__":
    main()
