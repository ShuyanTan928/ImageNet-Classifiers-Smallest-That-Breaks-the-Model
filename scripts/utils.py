"""Utility functions for loading models, datasets, and normalisation helpers."""
from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
import torchvision.transforms as T
from datasets import load_dataset
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models


_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class SampleMetadata:
    """Metadata describing a single ImageNet sample."""

    index: int
    label: int
    image_id: str


class HFImageNetDataset(Dataset):
    """Thin wrapper to apply torchvision transforms to HuggingFace datasets."""

    def __init__(self, hf_dataset, transform: Optional[T.Compose] = None):
        self._hf_dataset = hf_dataset
        self._transform = transform

    def __len__(self) -> int:
        return len(self._hf_dataset)

    def __getitem__(self, index: int):
        example = self._hf_dataset[int(index)]
        image = example["image"]
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        if self._transform is not None:
            image = self._transform(image)
        label = int(example["label"])
        image_id = example.get("image_id") or f"idx_{index:05d}"
        return image, label, image_id


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def imagenet_normalization() -> Tuple[torch.Tensor, torch.Tensor]:
    mean = torch.tensor(_IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(_IMAGENET_STD).view(3, 1, 1)
    return mean, std


def build_transform(resize: int = 256, crop_size: int = 224) -> T.Compose:
    return T.Compose(
        [
            T.Resize(resize),
            T.CenterCrop(crop_size),
            T.ToTensor(),
        ]
    )


def load_imagenet_dataset(
    dataset_name: str,
    split: str = "train",
    num_samples: Optional[int] = None,
    seed: int = 42,
    transform: Optional[T.Compose] = None,
) -> Tuple[Dataset, List[int]]:
    """Load the ImageNet validation dataset from HuggingFace and subsample."""

    hf_dataset = load_dataset(dataset_name, split=split)
    indices = list(range(len(hf_dataset)))
    if num_samples is not None:
        rng = random.Random(seed)
        rng.shuffle(indices)
        indices = indices[:num_samples]
        hf_dataset = hf_dataset.select(indices)
    else:
        indices = list(range(len(hf_dataset)))
    dataset = HFImageNetDataset(hf_dataset, transform=transform)
    return dataset, indices


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 1,
    num_workers: int = 0,
    shuffle: bool = False,
) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def load_model(model_name: str = "resnet18", device: Optional[torch.device] = None) -> torch.nn.Module:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model_name.lower() == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    model.eval()
    model.to(device)
    return model


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_json(path: str, data) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def batch_to_device(batch, device: torch.device):
    images, labels, image_ids = batch
    images = images.to(device)
    labels = labels.to(device)
    return images, labels, image_ids
