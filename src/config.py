import argparse
from pathlib import Path


def get_args() -> argparse.Namespace:
    """Parse and return the command-line arguments for the configuration.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Configuration.")

    parser.add_argument(
        "--data_path",
        type=str,
        default=str(Path(__file__).resolve().parent.parent / "data/iqa"),
        help="Path to the data directory",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for training"
    )
    parser.add_argument(
        "--num_workers", type=int, default=11, help="Number of data loading workers"
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of training epochs"
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="CADq",
        help="Weights & Biases project name",
    )
    parser.add_argument("--tag", type=str, default="baseline", help="Experiment tag")
    parser.add_argument("--head", type=str, default="feature_study", help="Head type)")
    parser.add_argument(
        "--dropout", type=float, default=0.0, help="Dropout rate (0.0 for no dropout)"
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default="cosine",
        choices=["cosine", "none"],
        help="Learning rate scheduler ('cosine', 'none')",
    )
    parser.add_argument(
        "--feature_level",
        type=int,
        default=4,
        choices=[1, 2, 3, 4, 5],
        help="Feature level (1, 2, 3, 4, or 5 for all)",
    )
    parser.add_argument(
        "--seed", type=int, default=15, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--freeze",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Freeze backbone weights during training",
    )
    parser.add_argument(
        "--final_training",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Perform 5-fold cross-validation",
    )

    args = parser.parse_args()

    args.data_path = Path(args.data_path)

    return args
