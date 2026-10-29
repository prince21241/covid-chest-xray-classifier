"""Read and write the split manifest."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

MANIFEST_COLUMNS = [
    "relative_path",
    "label",
    "label_name",
    "split",
    "sha256",
    "group_id",
    "width",
    "height",
    "patient_id",
    "exclude_reason",
]


def write_manifest(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(rows, key=lambda row: row["relative_path"])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in ordered:
            writer.writerow({column: row.get(column, "") for column in MANIFEST_COLUMNS})


def load_manifest(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = [column for column in MANIFEST_COLUMNS if column not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"Manifest is missing columns: {missing}")
        rows = []
        for row in reader:
            row["label"] = int(row["label"])
            row["width"] = int(row["width"])
            row["height"] = int(row["height"])
            rows.append(row)
    return rows


def rows_for_split(rows: list[dict], split: str) -> list[dict]:
    return [row for row in rows if row["split"] == split]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))
