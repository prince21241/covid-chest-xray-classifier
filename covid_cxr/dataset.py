"""PyTorch dataset for rows in the split manifest."""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import DataLoader, Dataset

from covid_cxr.engine import seed_worker


class ChestXrayDataset(Dataset):
    def __init__(self, rows: list[dict], dataset_root: str | Path, transform) -> None:
        self.rows = list(rows)
        self.dataset_root = Path(dataset_root)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        path = self.dataset_root / row["relative_path"]
        try:
            with Image.open(path) as image:
                rgb = image.convert("RGB")
        except (OSError, UnidentifiedImageError) as exc:
            raise RuntimeError(f"Could not read X-ray image: {path}") from exc
        features = self.transform(rgb)
        label = int(row["label"])
        return features, label


def make_loader(rows, dataset_root, transform, batch_size: int, shuffle: bool, seed: int, num_workers: int) -> DataLoader:
    dataset = ChestXrayDataset(rows, dataset_root, transform)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        generator=generator if shuffle else None,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker if num_workers else None,
    )
