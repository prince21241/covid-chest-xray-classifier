"""Train the ResNet-18 head, then optionally the final residual block.

Full training starts only with ``--full``. ``--smoke-test`` checks that data,
loss, and checkpoint saving work on a few images. The test split is not loaded.
"""

from __future__ import annotations

import csv
import math
import random
from datetime import datetime, timezone
from pathlib import Path

import torch
from sklearn.metrics import roc_auc_score
from torch import nn

from covid_cxr.config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_FINETUNE_EPOCHS,
    DEFAULT_LR,
    DEFAULT_LR_FINETUNE,
    DEFAULT_MIN_DELTA,
    DEFAULT_NUM_WORKERS,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PATIENCE,
    DEFAULT_SEED,
    DISCLAIMER,
    SMOKE_BATCH_SIZE,
    SMOKE_EPOCHS,
    SMOKE_TRAIN_PER_CLASS,
    SMOKE_VAL_PER_CLASS,
)
from covid_cxr.dataset import make_loader
from covid_cxr.discovery import resolve_dataset_root
from covid_cxr.engine import collect_predictions, describe_device, get_device, set_seed, train_one_epoch
from covid_cxr.manifest import file_sha256, load_manifest, read_json, rows_for_split
from covid_cxr.model import build_model, save_checkpoint, set_phase, trainable_parameter_count
from covid_cxr.plots import plot_training_curves
from covid_cxr.preprocessing import build_transforms


def _positive_weight(rows: list[dict]) -> float:
    n_covid = sum(int(row["label"]) == 1 for row in rows)
    n_normal = sum(int(row["label"]) == 0 for row in rows)
    if n_covid == 0 or n_normal == 0:
        raise RuntimeError("The training split needs both COVID and Normal images.")
    return n_normal / n_covid


def _sample_per_class(rows: list[dict], per_class: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    chosen: list[dict] = []
    for label in (0, 1):
        pool = [row for row in rows if int(row["label"]) == label]
        if len(pool) < per_class:
            raise RuntimeError(
                f"Smoke test needs at least {per_class} images with label {label} in this split, "
                f"but found {len(pool)}."
            )
        rng.shuffle(pool)
        chosen.extend(pool[:per_class])
    return chosen


def _dataset_root(data_dir: str | None, report: dict) -> Path:
    if data_dir:
        return resolve_dataset_root(data_dir)
    saved = Path(report["dataset_root"])
    if saved.exists():
        return saved
    raise RuntimeError(
        "The dataset path saved with the split no longer exists. "
        "Pass --data-dir pointing at the COVID and Normal folders."
    )


def _write_history(path: Path, history: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["phase", "epoch", "train_loss", "val_loss", "val_roc_auc", "learning_rate"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in history:
            writer.writerow(
                {
                    "phase": row["phase"],
                    "epoch": row["epoch"],
                    "train_loss": row["train_loss"],
                    "val_loss": row["val_loss"],
                    "val_roc_auc": "" if row["val_roc_auc"] is None else row["val_roc_auc"],
                    "learning_rate": row["learning_rate"],
                }
            )


def _validation_auc(labels, scores) -> float | None:
    if len(set(int(label) for label in labels)) < 2:
        return None
    return float(roc_auc_score(labels, scores))


def train(
    data_dir: str | None = None,
    output_dir: str | Path | None = None,
    seed: int = DEFAULT_SEED,
    batch_size: int = DEFAULT_BATCH_SIZE,
    lr: float = DEFAULT_LR,
    lr_finetune: float = DEFAULT_LR_FINETUNE,
    epochs: int = DEFAULT_EPOCHS,
    finetune_epochs: int = DEFAULT_FINETUNE_EPOCHS,
    patience: int = DEFAULT_PATIENCE,
    num_workers: int = DEFAULT_NUM_WORKERS,
    head_only: bool = False,
    smoke_test: bool = False,
    full: bool = False,
) -> Path:
    if smoke_test and full:
        raise RuntimeError("Choose either --smoke-test or --full.")
    if not smoke_test and not full:
        raise RuntimeError("Training was not started. Pass --smoke-test or --full.")
    if min(batch_size, epochs, patience) < 1:
        raise RuntimeError("Batch size, epochs, and patience must be at least 1.")
    if lr <= 0 or lr_finetune <= 0:
        raise RuntimeError("Learning rates must be positive.")
    if not head_only and not smoke_test and lr_finetune >= lr:
        print("WARNING: the fine-tuning learning rate is not smaller than the head learning rate.")

    output = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR
    manifest_path = output / "splits" / "manifest.csv"
    report_path = output / "splits" / "split_report.json"
    if not manifest_path.is_file() or not report_path.is_file():
        raise RuntimeError(
            "No split manifest found. Create one first:\n"
            "  python -m covid_cxr prepare"
        )

    report = read_json(report_path)
    dataset_root = _dataset_root(data_dir, report)
    rows = load_manifest(manifest_path)
    train_rows = rows_for_split(rows, "train")
    val_rows = rows_for_split(rows, "val")
    if not train_rows or not val_rows:
        raise RuntimeError("The manifest has no training or validation rows. Run prepare again.")

    if smoke_test:
        train_rows = _sample_per_class(train_rows, SMOKE_TRAIN_PER_CLASS, seed)
        val_rows = _sample_per_class(val_rows, SMOKE_VAL_PER_CLASS, seed + 1)
        batch_size = min(batch_size, SMOKE_BATCH_SIZE)
        epochs = SMOKE_EPOCHS
        head_only = True
        run_dir = output / "smoke"
    else:
        run_dir = output

    set_seed(seed)
    device = get_device()
    pos_weight_value = _positive_weight(train_rows)
    n_covid = sum(int(row["label"]) == 1 for row in train_rows)
    n_normal = sum(int(row["label"]) == 0 for row in train_rows)
    print(DISCLAIMER)
    print(f"Using device: {describe_device(device)}")
    print(f"Training images: COVID={n_covid}  Normal={n_normal}")
    if smoke_test:
        weight_note = "from the smoke-test sample, not the full training split"
    else:
        weight_note = "from the training split only"
    print(f"Positive-class weight (Normal/COVID), {weight_note}: {pos_weight_value:.4f}")
    if smoke_test:
        print("Smoke test: one head-only epoch on a small sample. This does not fit the full model.")
    elif report["patient_identifiers"].get("patient_independence_verified"):
        print("Patient identifiers were used to keep each patient in one split.")
    else:
        print("WARNING: " + report["patient_identifiers"]["message"])

    train_loader = make_loader(
        train_rows, dataset_root, build_transforms(train=True), batch_size, True, seed, num_workers
    )
    val_loader = make_loader(
        val_rows, dataset_root, build_transforms(train=False), batch_size, False, seed, num_workers
    )
    try:
        model = build_model(pretrained=True)
    except Exception as exc:
        raise RuntimeError(
            "Could not load ImageNet-pretrained ResNet-18 weights. "
            "Check the network connection and try again. The original error was: "
            f"{exc}"
        ) from exc
    model.to(device)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight_value], dtype=torch.float32, device=device)
    )

    phases = [("head", epochs, lr)]
    if not head_only:
        phases.append(("layer4", finetune_epochs, lr_finetune))

    best_loss = math.inf
    best_state = None
    best_phase = None
    history: list[dict] = []
    checkpoint_path = run_dir / "checkpoints" / "best.pt"
    history_path = run_dir / "metrics" / "history.csv"
    curve_path = run_dir / "plots" / "training_curves.png"
    manifest_hash = file_sha256(manifest_path)

    for phase, phase_epochs, phase_lr in phases:
        if best_state is not None:
            model.load_state_dict(best_state)
        set_phase(model, phase)
        optimizer = torch.optim.Adam([parameter for parameter in model.parameters() if parameter.requires_grad], lr=phase_lr)
        print(
            f"Phase {phase}: trainable parameters={trainable_parameter_count(model)} "
            f"learning rate={phase_lr} max epochs={phase_epochs}"
        )
        stale = 0
        for epoch in range(1, phase_epochs + 1):
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_labels, val_scores = collect_predictions(model, val_loader, criterion, device)
            val_auc = _validation_auc(val_labels, val_scores)
            improved = math.isfinite(val_loss) and val_loss < best_loss - DEFAULT_MIN_DELTA
            if improved:
                best_loss = val_loss
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                best_phase = phase
                stale = 0
                save_checkpoint(
                    checkpoint_path,
                    model,
                    {
                        "smoke_test": smoke_test,
                        "best_val_loss": best_loss,
                        "best_phase": best_phase,
                        "seed": seed,
                        "manifest_sha256": manifest_hash,
                        "dataset_root": str(dataset_root.resolve()),
                        "pos_weight": pos_weight_value,
                        "trained_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "hyperparameters": {
                            "batch_size": batch_size,
                            "lr": lr,
                            "lr_finetune": lr_finetune,
                            "epochs": epochs,
                            "finetune_epochs": finetune_epochs,
                            "patience": patience,
                            "head_only": head_only,
                            "min_delta": DEFAULT_MIN_DELTA,
                        },
                    },
                )
            else:
                stale += 1
            history.append(
                {
                    "phase": phase,
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_roc_auc": val_auc,
                    "learning_rate": phase_lr,
                }
            )
            auc_text = "n/a" if val_auc is None else f"{val_auc:.4f}"
            print(
                f"  [{phase}] epoch {epoch}/{phase_epochs}  "
                f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
                f"val_roc_auc={auc_text}  checkpoint={'saved' if improved else 'kept previous'}"
            )
            _write_history(history_path, history)
            plot_training_curves(history, curve_path)
            if stale >= patience:
                print(f"Early stopping in the {phase} phase after {patience} epochs without improvement.")
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a finite validation loss, so no checkpoint was saved.")
    model.load_state_dict(best_state)
    print(f"Best validation loss: {best_loss:.4f} (phase: {best_phase})")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Training curves: {curve_path}")
    if smoke_test:
        print("Smoke test finished. Review the lines above, then start full training when you are ready:")
        print("  python -m covid_cxr train --full")
    else:
        print("Next, choose the decision threshold on the validation set and score the test set once:")
        print("  python -m covid_cxr evaluate")
    return checkpoint_path
