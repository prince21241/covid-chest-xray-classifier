"""One training epoch and one validation pass."""

from __future__ import annotations

import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from covid_cxr.model import trainable_batchnorm_in_train_mode


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def describe_device(device: torch.device) -> str:
    if device.type == "cuda":
        return f"cuda ({torch.cuda.get_device_name(0)})"
    return "cpu"


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def train_one_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, optimizer, device: torch.device) -> float:
    trainable_batchnorm_in_train_mode(model)
    total_loss = 0.0
    seen = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device).float()
        optimizer.zero_grad(set_to_none=True)
        logits = model(images).view(-1)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        batch = int(labels.shape[0])
        total_loss += float(loss.item()) * batch
        seen += batch
    if seen == 0:
        raise RuntimeError("The training loader did not yield any images.")
    return total_loss / seen


@torch.no_grad()
def collect_predictions(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device):
    """Return average loss, integer labels, and sigmoid model scores."""
    model.eval()
    total_loss = 0.0
    seen = 0
    labels_all = []
    scores_all = []
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device).float()
        logits = model(images).view(-1)
        loss = criterion(logits, labels)
        batch = int(labels.shape[0])
        total_loss += float(loss.item()) * batch
        seen += batch
        labels_all.append(labels.detach().cpu())
        scores_all.append(torch.sigmoid(logits).detach().cpu())
    if seen == 0:
        raise RuntimeError("The loader did not yield any images.")
    label_tensor = torch.cat(labels_all).to(dtype=torch.int64).numpy()
    score_tensor = torch.cat(scores_all).numpy()
    return total_loss / seen, label_tensor, score_tensor
