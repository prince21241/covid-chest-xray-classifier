"""Choose a validation threshold, freeze it, then score the test set once."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from covid_cxr.config import DEFAULT_BATCH_SIZE, DEFAULT_NUM_WORKERS, DEFAULT_OUTPUT_DIR, DISCLAIMER
from covid_cxr.dataset import make_loader
from covid_cxr.discovery import resolve_dataset_root
from covid_cxr.engine import collect_predictions, describe_device, get_device
from covid_cxr.manifest import file_sha256, load_manifest, read_json, rows_for_split, write_json
from covid_cxr.metrics import THRESHOLD_RULE, always_normal_baseline, binary_metrics, select_threshold
from covid_cxr.model import load_checkpoint, model_from_checkpoint
from covid_cxr.plots import plot_confusion_matrix
from covid_cxr.preprocessing import build_transforms


def _print_block(title: str, metrics: dict) -> None:
    print(title)
    print(f"  images={metrics['n']}  COVID={metrics['n_covid']}  Normal={metrics['n_normal']}")
    for key in ("accuracy", "sensitivity", "specificity", "precision", "f1", "roc_auc", "pr_auc"):
        value = metrics[key]
        shown = "n/a" if value is None else f"{value:.4f}"
        print(f"  {key:12} {shown}")
    matrix = metrics["confusion_matrix"]
    print("  confusion matrix, rows true [Normal, COVID], columns predicted [Normal, COVID]:")
    print(f"    {matrix[0]}")
    print(f"    {matrix[1]}")


def evaluate(
    data_dir: str | None = None,
    output_dir: str | Path | None = None,
    checkpoint: str | Path | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    num_workers: int = DEFAULT_NUM_WORKERS,
) -> dict:
    output = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR
    checkpoint_path = Path(checkpoint) if checkpoint else output / "checkpoints" / "best.pt"
    if not checkpoint_path.is_file():
        raise RuntimeError(
            f"No checkpoint at {checkpoint_path}. Train first with:\n"
            "  python -m covid_cxr train --full"
        )
    saved = load_checkpoint(checkpoint_path)
    if saved.get("smoke_test"):
        raise RuntimeError(
            "This checkpoint is from the smoke test. It is not the model to evaluate. "
            "Run python -m covid_cxr train --full first."
        )

    manifest_path = output / "splits" / "manifest.csv"
    report_path = output / "splits" / "split_report.json"
    if not manifest_path.is_file() or not report_path.is_file():
        raise RuntimeError("The split manifest is missing. Run python -m covid_cxr prepare")
    current_hash = file_sha256(manifest_path)
    trained_hash = saved.get("manifest_sha256")
    if trained_hash and trained_hash != current_hash:
        raise RuntimeError(
            "The split manifest changed after this checkpoint was trained. "
            "Evaluation would mix a model with a different split. "
            "Train again, or point --checkpoint at a model from this manifest."
        )

    report = read_json(report_path)
    if data_dir:
        dataset_root = resolve_dataset_root(data_dir)
    else:
        dataset_root = Path(report["dataset_root"])
        if not dataset_root.exists():
            raise RuntimeError("The saved dataset path does not exist. Pass --data-dir.")

    rows = load_manifest(manifest_path)
    val_rows = rows_for_split(rows, "val")
    if not val_rows:
        raise RuntimeError("The validation split is empty.")

    device = get_device()
    print(DISCLAIMER)
    print(f"Using device: {describe_device(device)}")
    print("Selecting the decision threshold from the validation set only.")
    model = model_from_checkpoint(saved)
    model.to(device)
    pos_weight = float(saved.get("pos_weight", 1.0))
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], dtype=torch.float32, device=device)
    )
    eval_transform = build_transforms(train=False, config=saved["preprocessing"])
    val_loader = make_loader(val_rows, dataset_root, eval_transform, batch_size, False, int(saved.get("seed", 0)), num_workers)
    _, val_labels, val_scores = collect_predictions(model, val_loader, criterion, device)
    threshold, validation_metrics = select_threshold(val_labels, val_scores)
    print(f"Frozen validation threshold: {threshold:.2f}")
    print(THRESHOLD_RULE)

    saved["threshold"] = threshold
    saved["threshold_source"] = "validation_max_f1"
    saved["threshold_rule"] = THRESHOLD_RULE
    torch.save(saved, checkpoint_path)

    test_rows = rows_for_split(rows, "test")
    if not test_rows:
        raise RuntimeError("The test split is empty, so there is nothing to evaluate.")
    test_loader = make_loader(
        test_rows, dataset_root, eval_transform, batch_size, False, int(saved.get("seed", 0)), num_workers
    )
    _, test_labels, test_scores = collect_predictions(model, test_loader, criterion, device)
    test_metrics = binary_metrics(test_labels, test_scores, threshold)
    baseline = always_normal_baseline(test_labels)

    payload = {
        "disclaimer": DISCLAIMER,
        "positive_class": "COVID",
        "negative_class": "Normal",
        "threshold": threshold,
        "threshold_rule": THRESHOLD_RULE,
        "threshold_source": "validation",
        "test_was_used_for_threshold": False,
        "checkpoint": str(checkpoint_path.resolve()),
        "validation": validation_metrics,
        "test": test_metrics,
        "always_normal_baseline": baseline,
        "notes": [
            "Validation metrics show the threshold that was selected. They are not a final score.",
            "Test metrics were computed once after the threshold was frozen.",
            "The always-Normal baseline predicts the majority class for every test image.",
            "These figures are experimental results on this split. They are not clinical accuracy.",
        ],
    }
    metrics_path = output / "metrics" / "test_metrics.json"
    figure_path = output / "plots" / "confusion_matrix_test.png"
    write_json(metrics_path, payload)
    plot_confusion_matrix(test_metrics["confusion_matrix"], figure_path, "Test set, threshold frozen on validation")

    _print_block("Validation at the frozen threshold (selection set):", validation_metrics)
    _print_block("Test at the frozen threshold:", test_metrics)
    _print_block("Always-Normal baseline on the test set:", baseline)
    print(f"Metrics: {metrics_path}")
    print(f"Confusion matrix plot: {figure_path}")
    print("The test set was scored once and was not used to choose the threshold.")
    return payload
