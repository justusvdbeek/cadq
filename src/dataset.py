import json
import logging
import re
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from preprocess import find_roi


def _determine_manufacturer(annotation_file: str) -> str:
    """Determine the manufacturer based on the annotation file name."""
    lower_name = annotation_file.lower()
    if "bl7000eg760" in lower_name or "ep8000" in lower_name or "ep7000" in lower_name or "fujifilm" in lower_name:
        return "fuji"
    if "x1ez1500" in lower_name or "olympus" in lower_name or "x1hq190" in lower_name:
        return "olympus"
    if "i8020ci20" in lower_name:
        return "pentax"
    if "_na_" in lower_name:
        return "unknown"
    logging.warning("Unknown manufacturer for %s", annotation_file)
    return "unknown"


def _load_json(path: Path) -> dict | None:
    """Load JSON data from the specified path."""
    try:
        with path.open(encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        logging.exception("Failed to read/parse JSON: %s", path)
        return None


def _validate_dirs(ann_dir: Path, img_dir: Path) -> None:
    """Validate that the annotation and image directories exist."""
    if not ann_dir.is_dir():
        error_message = f"Annotations directory not found: {ann_dir}"
        raise FileNotFoundError(error_message)
    if not img_dir.is_dir():
        error_message = f"Images directory not found: {img_dir}"
        raise FileNotFoundError(error_message)


def _resolve_image_path(images_dir: Path, annotation: dict, ann_name: str) -> tuple[str | None, Path | None]:
    """Resolve the image file path from the annotation data."""
    file_path = annotation.get("file_path")
    if not file_path:
        logging.warning("Missing file_path in annotation: %s", ann_name)
        return None, None
    image_file_name = Path(file_path).name
    image_path = images_dir / image_file_name
    if not image_path.is_file():
        logging.warning("Image file not found for %s -> %s", ann_name, image_path)
        return None, None
    return image_file_name, image_path


def _extract_labels_and_annotators(annotation: dict) -> tuple[dict, set[str]] | None:
    """Extract labels and annotators from the annotation data."""
    label_map = {"Poor": 0, "Adequate": 1, "Good": 2}
    labels = {
        "Rate mucosal cleaning": np.nan,
        "Rate expansion": np.nan,
        "Rate OIQ": np.nan,
    }
    annotators: set[str] = set()

    anns = annotation.get("annotations", [])
    if not isinstance(anns, list):
        return None

    for ann in anns:
        job_desc = ann.get("job_description")
        choice = ann.get("choice_name")
        if job_desc in labels and choice in label_map:
            labels[job_desc] = label_map[choice]
            full_name = (ann.get("annotator") or {}).get("full_name")
            if full_name:
                annotators.add(full_name)

    return labels, annotators


def _parse_patient_str(image_file_name: str) -> str | None:
    """Parse the patient string from the image file name."""
    stem = Path(image_file_name).stem
    parts = stem.split("_")
    min_parts_length = 2
    if len(parts) < min_parts_length:
        return None
    nums = re.findall(r"\d+", parts[1])
    return f"{parts[0]}_{nums[0]}" if nums else None


def _apply_exclusions(
    oiq: float, clean: float, exp: float, retro: float, *, exclusion: bool = False
) -> tuple[float, float, float, float]:
    """Apply exclusion rules to the labels based on the specified criteria."""
    if not exclusion:
        return oiq, clean, exp, retro
    if oiq == 0:
        return oiq, np.nan, np.nan, np.nan
    if retro == 1:
        return oiq, clean, np.nan, retro
    return oiq, clean, exp, retro


def _apply_binary(
    oiq: float, clean: float, exp: float, retro: float, *, binary: bool = False
) -> tuple[float, float, float, float]:
    """Convert labels to binary format if specified."""
    if not binary:
        return oiq, clean, exp, retro

    def remap(value: float | None) -> float:
        return np.nan if pd.isna(value) else (1.0 if int(value) in (1, 2) else 0.0)

    return remap(oiq), remap(clean), remap(exp), remap(retro)


def load_image_dataframe(
    data_path: str | Path, split: str = "train", *, exclusion: bool = True, binary: bool = False
) -> pd.DataFrame:
    """Load IQA DataFrame containing image paths and their corresponding labels.

    Args:
        data_path (str | Path): Path to the dataset directory.
        split (str): Dataset split to load (e.g., "train"). Defaults to "train".
        exclusion (bool): Whether to apply exclusion rules to the labels. Defaults to True.
        binary (bool): Whether to convert labels to binary format. Defaults to False.

    Returns:
        pd.DataFrame: A DataFrame containing image paths and their associated labels.
    """
    annotations_dir = Path(data_path) / split / "annotations/classification/Unassigned"
    images_dir = Path(data_path) / split / "data/Unassigned"
    _validate_dirs(annotations_dir, images_dir)

    data_records: list[dict] = []
    patient_id_map: dict[str, int] = {}
    next_patient_id = 0

    for annotation_path in annotations_dir.glob("*.json"):
        annotation_filename = annotation_path.name
        manufacturer = _determine_manufacturer(annotation_filename)

        annotation_data = _load_json(annotation_path)
        if annotation_data is None:
            continue

        image_file_name, image_path = _resolve_image_path(images_dir, annotation_data, annotation_filename)
        if image_file_name is None or image_path is None:
            continue

        patient_str = _parse_patient_str(image_file_name)
        if patient_str is None:
            logging.warning("Could not parse patient string from filename: %s", image_file_name)
            continue

        patient_id = patient_id_map.setdefault(patient_str, next_patient_id)
        if patient_id == next_patient_id:
            next_patient_id += 1

        extracted = _extract_labels_and_annotators(annotation_data)
        if extracted is None:
            logging.warning("Invalid annotations format in file: %s", annotation_filename)
            continue
        labels, annotators = extracted

        if all(pd.isna(v) for v in labels.values()):
            logging.warning("All labels missing in file '%s'", annotation_filename)
            continue

        oiq = labels["Rate OIQ"]
        clean = labels["Rate mucosal cleaning"]
        exp = labels["Rate expansion"]
        retro = int("retro" in image_file_name.lower())

        oiq, clean, exp, retro = _apply_exclusions(oiq, clean, exp, retro, exclusion=exclusion)
        oiq, clean, exp, retro = _apply_binary(oiq, clean, exp, retro, binary=binary)

        if pd.isna(oiq):
            continue

        data_records.append(
            {
                "image_path": str(image_path),
                "oiq": oiq,
                "mucosal_cleaning": clean,
                "expansion": exp,
                "retrograde": retro,
                "patient_id": patient_id,
                "patient_str": patient_str,
                "manufacturer": manufacturer,
                "annotators": ",".join(sorted(annotators)),
            }
        )

    dataframe = pd.DataFrame(data_records)

    return dataframe


class ImageDataset(Dataset):
    """Dataset for IQA images and labels."""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        transform: Callable | None = None,
        *,
        preprocess: bool = True,
    ) -> None:
        self.transform = transform
        self.dataframe = dataframe
        self.preprocess = preprocess
        self.ignore_index = -100

    def _preprocess_image(self, image_path: str) -> Image.Image:
        """Preprocess the image by extracting the ROI."""
        try:
            pil_image = Image.open(image_path).convert("RGB")
        except Exception:
            logging.exception("Failed to open image: %s", image_path)
            raise

        try:
            rmin, rmax, cmin, cmax = find_roi(pil_image)
            return pil_image.crop((cmin, rmin, cmax, rmax))
        except Exception:
            logging.exception("ROI extraction failed for %s", image_path)
            return pil_image

    def _safe_label(self, val: float | None) -> torch.Tensor:
        """Convert label to tensor, using ignore_index for NaN values."""
        return torch.tensor(self.ignore_index if pd.isna(val) else int(val), dtype=torch.long)

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.dataframe)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Retrieve the dataset sample at the specified index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            A tuple containing the image tensor and its corresponding labels.
        """
        row = self.dataframe.iloc[idx]

        image = (
            self._preprocess_image(row["image_path"])
            if self.preprocess
            else Image.open(row["image_path"]).convert("RGB")
        )

        clean = self._safe_label(row["mucosal_cleaning"])
        expansion = self._safe_label(row["expansion"])
        oiq = self._safe_label(row["oiq"])
        retro = self._safe_label(row["retrograde"])

        if self.transform:
            image = self.transform(image)

        return image, clean, expansion, oiq, retro
