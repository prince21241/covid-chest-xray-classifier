"""Validate images, check duplicates and metadata, and write the split manifest."""

from __future__ import annotations

import hashlib
import io
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from covid_cxr import __version__
from covid_cxr.config import CLASS_TO_INDEX, DEFAULT_SEED, DISCLAIMER
from covid_cxr.discovery import (
    count_mask_files,
    find_named_child,
    ignored_directories,
    list_xray_images,
    resolve_dataset_root,
)
from covid_cxr.manifest import load_manifest, read_json, write_json, write_manifest
from covid_cxr.metadata import (
    assess_patient_identifiers,
    find_metadata_files,
    patient_id_lookup,
    read_metadata_workbook,
)
from covid_cxr.splits import RATIOS, assign_splits, group_records


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _inspect_image(path: Path) -> tuple[str, int, int]:
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    try:
        with Image.open(io.BytesIO(data)) as image:
            image.load()
            width, height = image.size
    except (OSError, UnidentifiedImageError, ValueError) as exc:
        raise OSError(str(exc)) from exc
    if width < 1 or height < 1:
        raise OSError("image has no pixels")
    return digest, int(width), int(height)


def _parse_size(text: str) -> tuple[int, int] | None:
    cleaned = text.lower().replace("×", "*").replace("x", "*")
    parts = [part.strip() for part in cleaned.split("*")]
    if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
        return int(parts[0]), int(parts[1])
    return None


def _split_summary(rows: list[dict]) -> dict:
    summary = {}
    for split in ("train", "val", "test", "excluded"):
        subset = [row for row in rows if row["split"] == split]
        summary[split] = {
            "COVID": sum(int(row["label"]) == CLASS_TO_INDEX["COVID"] for row in subset),
            "Normal": sum(int(row["label"]) == CLASS_TO_INDEX["Normal"] for row in subset),
            "total": len(subset),
        }
    return summary


def _print_summary(report: dict) -> None:
    print(DISCLAIMER)
    print(f"Dataset: {report['dataset_root']}")
    print(f"Classes used: {', '.join(report['classes_used'])}")
    if report["classes_ignored"]:
        print("Other folders were not used: " + ", ".join(report["classes_ignored"]))
    print(
        "Mask files were not used as model inputs: "
        + ", ".join(f"{name}={count}" for name, count in report["mask_files_skipped"].items())
    )
    print("Images opened:")
    for name, count in report["class_counts_valid"].items():
        print(f"  {name}: {count}")
    print(f"Invalid files: {len(report['invalid_files'])}")
    for item in report["invalid_files"][:10]:
        print(f"  {item['relative_path']}: {item['error']}")
    duplicates = report["duplicates"]
    print(
        "Exact duplicate groups (same label): "
        f"{duplicates['same_label_groups']} "
        f"({duplicates['extra_copies']} extra copies kept with their group)"
    )
    print(f"Duplicate groups with conflicting labels, excluded: {duplicates['conflicting_groups']}")
    print("WARNING: " + report["patient_identifiers"]["message"])
    print("Split image counts (group isolation overrides exact 70/15/15 proportions):")
    usable = sum(report["split_counts"][name]["total"] for name in ("train", "val", "test"))
    for name in ("train", "val", "test", "excluded"):
        counts = report["split_counts"][name]
        share = ""
        if name != "excluded" and usable:
            share = f"  {100.0 * counts['total'] / usable:.1f}% of usable images"
        print(
            f"  {name:8} COVID={counts['COVID']:5d}  Normal={counts['Normal']:5d}  "
            f"total={counts['total']:5d}{share}"
        )
    if report.get("metadata_size_mismatches"):
        print(
            "Metadata SIZE disagrees with the opened image for "
            f"{report['metadata_size_mismatches']} files. Opened sizes were used."
        )
    print("Opened image sizes: " + ", ".join(
        f"{size}={count}" for size, count in report["opened_image_sizes"].items()
    ))
    print(f"Manifest: {report['manifest_path']}")


def _listing_fingerprint(paths: list[Path]) -> dict:
    return {
        "n_listed": len(paths),
        "total_bytes": sum(path.stat().st_size for path in paths),
    }


def prepare(data_dir: str | None = None, output_dir: str | Path | None = None, seed: int = DEFAULT_SEED, resplit: bool = False) -> dict:
    from covid_cxr.config import DEFAULT_OUTPUT_DIR

    dataset_root = resolve_dataset_root(data_dir)
    output = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR
    split_dir = output / "splits"
    manifest_path = split_dir / "manifest.csv"
    report_path = split_dir / "split_report.json"

    listed: dict[str, list[Path]] = {}
    mask_counts: dict[str, int] = {}
    all_paths: list[Path] = []
    for class_name in ("COVID", "Normal"):
        class_dir = find_named_child(dataset_root, class_name)
        if class_dir is None:
            raise RuntimeError(f"Missing class directory: {class_name}")
        paths = list_xray_images(class_dir)
        listed[class_name] = paths
        all_paths.extend(paths)
        mask_counts[class_name] = count_mask_files(class_dir)
        if not paths:
            raise RuntimeError(f"No X-ray images found for {class_name} in {class_dir}")

    fingerprint = _listing_fingerprint(all_paths)
    if manifest_path.is_file() and report_path.is_file() and not resplit:
        existing = read_json(report_path)
        same_root = Path(existing.get("dataset_root", "")).resolve() == dataset_root.resolve()
        same_seed = int(existing.get("seed", -1)) == int(seed)
        same_files = existing.get("listing") == fingerprint
        saved_ratios = existing.get("ratios", {})
        same_ratios = all(abs(float(saved_ratios.get(name, -1)) - RATIOS[name]) < 1e-9 for name in RATIOS)
        if same_root and same_seed and same_files and same_ratios:
            rows = load_manifest(manifest_path)
            sample = [row for row in rows if row["split"] == "train"][:3]
            missing = [row["relative_path"] for row in sample if not (dataset_root / row["relative_path"]).is_file()]
            if missing:
                raise RuntimeError(
                    "The saved manifest does not match files under "
                    f"{dataset_root}. Rebuild it with --resplit."
                )
            print(f"Reusing split manifest: {manifest_path}")
            print("Run prepare again with --resplit to rebuild it.")
            print("WARNING: " + existing["patient_identifiers"]["message"])
            existing["reused"] = True
            return existing
        print("A split manifest already exists and does not match this dataset, seed, or file listing.")
        print("Run again with --resplit to replace it.")
        raise SystemExit(2)

    print(f"Checking {len(all_paths)} X-ray images under {dataset_root}")
    print("Opening and hashing every file. This can take several minutes.")
    records: list[dict] = []
    invalid: list[dict] = []
    for class_name in ("COVID", "Normal"):
        for path in listed[class_name]:
            relative = path.relative_to(dataset_root).as_posix()
            if any(part.lower() in {"mask", "masks"} for part in Path(relative).parts):
                raise RuntimeError(f"Mask path was listed as an X-ray: {relative}")
            try:
                digest, width, height = _inspect_image(path)
            except OSError as exc:
                invalid.append({"relative_path": relative, "label_name": class_name, "error": str(exc)})
                continue
            records.append(
                {
                    "relative_path": relative,
                    "label": CLASS_TO_INDEX[class_name],
                    "label_name": class_name,
                    "sha256": digest,
                    "width": width,
                    "height": height,
                    "patient_id": "",
                }
            )
            done = len(records) + len(invalid)
            if done % 1000 == 0 or done == len(all_paths):
                print(f"  opened {done}/{len(all_paths)}")

    valid_counts = Counter(record["label_name"] for record in records)
    for class_name in ("COVID", "Normal"):
        if valid_counts[class_name] == 0:
            raise RuntimeError(f"Every {class_name} image failed to open. Nothing to split.")

    tables = []
    metadata_files = find_metadata_files(dataset_root)
    for class_name, path in metadata_files.items():
        columns, rows = read_metadata_workbook(path)
        tables.append({"class_name": class_name, "path": str(path), "columns": columns, "rows": rows})
    patient_info = assess_patient_identifiers(tables)

    metadata_by_stem: dict[tuple[str, str], dict] = {}
    for table in tables:
        for row in table["rows"]:
            file_name = ""
            for key, value in row.items():
                if key.strip().lower() in {"file name", "filename", "file_name"}:
                    file_name = str(value).strip()
                    break
            if file_name:
                metadata_by_stem[(table["class_name"].lower(), Path(file_name).stem.lower())] = row

    size_mismatches = 0
    for record in records:
        meta = metadata_by_stem.get((record["label_name"].lower(), Path(record["relative_path"]).stem.lower()))
        if not meta:
            continue
        size_text = ""
        for key, value in meta.items():
            if key.strip().lower() == "size":
                size_text = str(value)
                break
        parsed = _parse_size(size_text) if size_text else None
        if parsed and parsed != (record["width"], record["height"]):
            size_mismatches += 1

    use_patient_ids = bool(patient_info["available"] and patient_info["identifier_column"])
    if use_patient_ids:
        lookup = patient_id_lookup(tables, patient_info["identifier_column"])
        matched = 0
        for record in records:
            key = (record["label_name"].lower(), Path(record["relative_path"]).stem.lower())
            record["patient_id"] = lookup.get(key, "")
            if record["patient_id"]:
                matched += 1
        coverage = matched / len(records)
        unique_ids = {record["patient_id"] for record in records if record["patient_id"]}
        most_common = Counter(record["patient_id"] for record in records if record["patient_id"]).most_common(1)
        too_common = bool(most_common and most_common[0][1] > 0.5 * len(records))
        if coverage < 0.95 or len(unique_ids) < 2 or too_common:
            use_patient_ids = False
            for record in records:
                record["patient_id"] = ""
            patient_info["available"] = False
            patient_info["patient_independence_verified"] = False
            patient_info["identifier_column"] = None
            patient_info["message"] = (
                "Patient independence cannot be verified. A column looked like a patient "
                "identifier, but it did not cover the images reliably enough to split on. "
                "Filenames were not treated as patient IDs."
            )
    patient_info["used_for_split"] = use_patient_ids

    groups = group_records(records, use_patient_ids=use_patient_ids)
    assignment = assign_splits(groups, seed=seed)
    manifest_rows: list[dict] = []
    for group in groups:
        if group["conflict"]:
            split = "excluded"
            reason = "conflicting_labels"
        else:
            split = assignment[group["group_id"]]
            reason = ""
        for record in group["records"]:
            manifest_rows.append(
                {
                    **record,
                    "group_id": group["group_id"],
                    "split": split,
                    "exclude_reason": reason,
                    "patient_id": record.get("patient_id", ""),
                }
            )

    same_label_groups = [group for group in groups if not group["conflict"] and group["n"] > 1]
    conflict_groups = [group for group in groups if group["conflict"]]

    def group_example(group: dict) -> dict:
        return {
            "group_id": group["group_id"],
            "n": group["n"],
            "labels": sorted({record["label_name"] for record in group["records"]}),
            "paths": [record["relative_path"] for record in group["records"][:8]],
        }

    opened_sizes = Counter(f"{record['width']}x{record['height']}" for record in records)
    report = {
        "project_version": __version__,
        "created_at": _now(),
        "disclaimer": DISCLAIMER,
        "dataset_root": str(dataset_root.resolve()),
        "seed": int(seed),
        "ratios": RATIOS,
        "listing": fingerprint,
        "classes_used": ["COVID", "Normal"],
        "classes_ignored": ignored_directories(dataset_root),
        "masks_used_as_inputs": False,
        "mask_files_skipped": mask_counts,
        "class_counts_listed": {name: len(paths) for name, paths in listed.items()},
        "class_counts_valid": {name: int(valid_counts[name]) for name in ("COVID", "Normal")},
        "invalid_files": invalid,
        "opened_image_sizes": dict(opened_sizes),
        "metadata_files": {name: str(path) for name, path in metadata_files.items()},
        "metadata_size_mismatches": size_mismatches,
        "patient_identifiers": patient_info,
        "split_unit": "patient_and_duplicate_component" if use_patient_ids else "exact_duplicate_group",
        "duplicates": {
            "same_label_groups": len(same_label_groups),
            "extra_copies": int(sum(group["n"] - 1 for group in same_label_groups)),
            "conflicting_groups": len(conflict_groups),
            "conflicting_images": int(sum(group["n"] for group in conflict_groups)),
            "same_label_examples": [group_example(group) for group in same_label_groups[:20]],
            "conflicting_examples": [group_example(group) for group in conflict_groups[:20]],
        },
        "split_counts": _split_summary(manifest_rows),
        "manifest_path": str(manifest_path.resolve()),
        "python": sys.version.split()[0],
        "pillow": Image.__version__,
        "reused": False,
    }
    write_manifest(manifest_path, manifest_rows)
    write_json(report_path, report)
    _print_summary(report)
    train_counts = report["split_counts"]["train"]
    if train_counts["COVID"] == 0 or train_counts["Normal"] == 0:
        raise RuntimeError(
            "The training split is missing COVID or Normal images. "
            "See the manifest exclusions before training."
        )
    for split_name in ("val", "test"):
        if report["split_counts"][split_name]["total"] == 0:
            print(f"WARNING: the {split_name} split has no images.")
    return report
