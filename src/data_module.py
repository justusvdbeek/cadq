import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from dataset import ImageDataset
from transforms import get_transforms


class ImageDataModule(pl.LightningDataModule):
    """A PyTorch Lightning DataModule for handling endoscopy datasets.

    This module prepares the training, validation, and test datasets
    and provides corresponding dataloaders.
    """

    def __init__(
        self,
        dataframes: dict[str, pd.DataFrame],
        batch_size: int,
        num_workers: int,
    ) -> None:
        super().__init__()
        self.train_dataframe = dataframes.get("train")
        self.val_dataframe = dataframes.get("val")
        self.test_dataframe = dataframes.get("test")
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: str | None = None) -> None:
        """Set up datasets for training, validation, and testing.

        Args:
            stage (str, optional): The stage of the setup process. Can be "fit", "test", or None.
        """
        if stage == "fit" or stage is None:
            self.train_dataset = ImageDataset(
                dataframe=self.train_dataframe, transform=get_transforms(train=True), preprocess=True
            )
            self.val_dataset = ImageDataset(
                dataframe=self.val_dataframe, transform=get_transforms(train=False), preprocess=True
            )

        if stage == "test" or stage is None:
            self.test_dataset = ImageDataset(
                dataframe=self.test_dataframe, transform=get_transforms(train=False), preprocess=True
            )

    def train_dataloader(self) -> DataLoader:
        """Returns the DataLoader for the training dataset."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )

    def val_dataloader(self) -> DataLoader:
        """Returns the DataLoader for the validation dataset."""
        if self.val_dataframe is None:
            return []
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )

    def test_dataloader(self) -> DataLoader:
        """Returns the DataLoader for the test dataset."""
        if self.test_dataframe is None:
            return []
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )
