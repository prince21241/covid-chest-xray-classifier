"""Score one chest X-ray with the saved checkpoint and the evaluation transform."""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image, UnidentifiedImageError

from covid_cxr.config import DEFAULT_OUTPUT_DIR, DISCLAIMER
from covid_cxr.engine import describe_device, get_device
from covid_cxr.model import load_checkpoint, model_from_checkpoint
from covid_cxr.preprocessing import build_transforms


def predict(image: str | Path, checkpoint: str | Path | None = None, output_dir: str | Path | None = None) -> dict:
    image_path = Path(image)
    if not image_path.is_file():
        raise RuntimeError(f"Image not found: {image_path}")
    output = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR
    checkpoint_path = Path(checkpoint) if checkpoint else output / "checkpoints" / "best.pt"
    if not checkpoint_path.is_file():
        raise RuntimeError(
            f"No checkpoint at {checkpoint_path}. Train and evaluate first, or pass --checkpoint."
        )

    saved = load_checkpoint(checkpoint_path)
    if saved.get("smoke_test"):
        print("WARNING: this checkpoint is from the smoke test, not a full training run.")
    threshold = saved.get("threshold")
    if threshold is None:
        threshold = 0.5
        threshold_note = (
            "No validation threshold is stored yet. Using 0.5 until you run "
            "python -m covid_cxr evaluate."
        )
    else:
        threshold = float(threshold)
        threshold_note = f"Threshold source: {saved.get('threshold_source', 'checkpoint')}"

    device = get_device()
    model = model_from_checkpoint(saved)
    model.to(device)
    model.eval()
    transform = build_transforms(train=False, config=saved["preprocessing"])
    try:
        with Image.open(image_path) as opened:
            rgb = opened.convert("RGB")
    except (OSError, UnidentifiedImageError) as exc:
        raise RuntimeError(f"Could not open {image_path} as an image.") from exc

    features = transform(rgb).unsqueeze(0).to(device)
    with torch.no_grad():
        logit = model(features).view(-1)[0]
        score = float(torch.sigmoid(logit).detach().cpu())
    predicted = "COVID" if score >= threshold else "Normal"
    result = {
        "image": str(image_path.resolve()),
        "model_score_covid": score,
        "threshold": threshold,
        "predicted_label": predicted,
        "score_meaning": (
            "model score from the sigmoid of a ResNet-18 logit; "
            "not a calibrated probability of infection"
        ),
        "device": describe_device(device),
        "checkpoint": str(checkpoint_path.resolve()),
        "disclaimer": DISCLAIMER,
    }
    print(f"Image: {result['image']}")
    print(f"Model score (COVID): {score:.4f}")
    print(f"Decision threshold: {threshold:.2f}")
    print(threshold_note)
    print(f"Predicted label: {predicted}")
    print("This is a model score, not a calibrated probability of infection.")
    print(DISCLAIMER)
    return result
