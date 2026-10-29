"""Split, preprocessing, discovery, and metadata checks. No model is trained."""

from __future__ import annotations

import io
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from PIL import Image

from covid_cxr.config import PROJECT_ROOT
from covid_cxr.discovery import (
    count_mask_files,
    ignored_directories,
    list_xray_images,
    resolve_dataset_root,
    DatasetNotFoundError,
)
from covid_cxr.manifest import load_manifest
from covid_cxr.metadata import assess_patient_identifiers, read_metadata_workbook
from covid_cxr.metrics import always_normal_baseline, binary_metrics, select_threshold
from covid_cxr.prepare import prepare
from covid_cxr.preprocessing import ResizeWithPad, preprocessing_config
from covid_cxr.splits import assign_splits, group_records


def _record(path: str, label: int, digest: str, patient: str = "") -> dict:
    name = "COVID" if label == 1 else "Normal"
    return {
        "relative_path": path,
        "label": label,
        "label_name": name,
        "sha256": digest,
        "patient_id": patient,
    }


def _png_bytes(color: tuple[int, int, int], size: tuple[int, int] = (30, 18)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, format="PNG")
    return buffer.getvalue()


class SplitTests(unittest.TestCase):
    def test_singleton_split_is_stratified_and_reproducible(self) -> None:
        groups = []
        for index in range(100):
            groups.append({"group_id": f"neg-{index}", "label": 0, "n": 1, "conflict": False})
            groups.append({"group_id": f"pos-{index}", "label": 1, "n": 1, "conflict": False})
        first = assign_splits(groups, seed=42)
        second = assign_splits(groups, seed=42)
        self.assertEqual(first, second)
        counts = Counter(first.values())
        self.assertEqual(counts["train"], 140)
        self.assertEqual(counts["val"], 30)
        self.assertEqual(counts["test"], 30)
        for label, prefix in ((0, "neg"), (1, "pos")):
            owned = [split for group_id, split in first.items() if group_id.startswith(prefix)]
            owned_counts = Counter(owned)
            self.assertEqual(owned_counts["train"], 70)
            self.assertEqual(owned_counts["val"], 15)
            self.assertEqual(owned_counts["test"], 15)

    def test_a_group_is_assigned_once(self) -> None:
        groups = [{"group_id": "big", "label": 1, "n": 10, "conflict": False}]
        groups.extend({"group_id": f"s{index}", "label": 1, "n": 1, "conflict": False} for index in range(10))
        assignment = assign_splits(groups, seed=3)
        self.assertEqual(len(assignment), 11)
        self.assertIn(assignment["big"], {"train", "val", "test"})

    def test_duplicate_bytes_and_patients_stay_together(self) -> None:
        records = [
            _record("COVID/images/a.png", 1, "same", "p1"),
            _record("COVID/images/b.png", 1, "same", "p9"),
            _record("Normal/images/c.png", 0, "other", "p2"),
            _record("Normal/images/d.png", 0, "other2", "p2"),
        ]
        groups = group_records(records, use_patient_ids=True)
        by_path = {}
        for group in groups:
            for path in group["paths"]:
                by_path[path] = group["group_id"]
        self.assertEqual(by_path["COVID/images/a.png"], by_path["COVID/images/b.png"])
        self.assertEqual(by_path["Normal/images/c.png"], by_path["Normal/images/d.png"])
        self.assertTrue(all(not group["conflict"] for group in groups))

    def test_conflicting_labels_are_excluded_from_assignment(self) -> None:
        records = [
            _record("COVID/images/a.png", 1, "shared"),
            _record("Normal/images/a.png", 0, "shared"),
        ]
        groups = group_records(records, use_patient_ids=False)
        self.assertEqual(len(groups), 1)
        self.assertTrue(groups[0]["conflict"])
        self.assertEqual(assign_splits(groups, seed=1), {})

    def test_same_patient_with_two_labels_is_a_conflict(self) -> None:
        records = [
            _record("COVID/images/a.png", 1, "h1", "patient-7"),
            _record("Normal/images/b.png", 0, "h2", "patient-7"),
        ]
        groups = group_records(records, use_patient_ids=True)
        self.assertEqual(len(groups), 1)
        self.assertTrue(groups[0]["conflict"])


class PreprocessingTests(unittest.TestCase):
    def test_letterbox_keeps_the_long_side_and_pads_the_short_side(self) -> None:
        resize = ResizeWithPad(224, fill=0)
        wide = resize(Image.new("RGB", (40, 20), (255, 255, 255)))
        tall = resize(Image.new("L", (20, 40), 255))
        self.assertEqual(wide.size, (224, 224))
        self.assertEqual(wide.mode, "RGB")
        self.assertEqual(tall.mode, "RGB")
        wide_box = _content_box(wide)
        tall_box = _content_box(tall)
        self.assertEqual(wide_box[0], 0)
        self.assertEqual(wide_box[2], 223)
        self.assertGreater(wide_box[1], 0)
        self.assertLess(wide_box[3], 223)
        self.assertEqual(tall_box[1], 0)
        self.assertEqual(tall_box[3], 223)
        self.assertGreater(tall_box[0], 0)
        self.assertLess(tall_box[2], 223)

    def test_saved_preprocessing_uses_imagenet_statistics_and_mild_augmentation(self) -> None:
        config = preprocessing_config()
        self.assertEqual(config["image_size"], 224)
        self.assertEqual(config["imagenet_mean"], [0.485, 0.456, 0.406])
        self.assertEqual(config["imagenet_std"], [0.229, 0.224, 0.225])
        augmentation = config["train_augmentation"]
        self.assertEqual(augmentation["rotation_degrees"], 10)
        self.assertEqual(augmentation["translate_fraction"], 0.05)
        self.assertFalse(augmentation["horizontal_flip"])
        self.assertFalse(augmentation["vertical_flip"])
        self.assertFalse(augmentation["random_resized_crop"])
        self.assertIsNone(config["evaluation_augmentation"])


class MetricTests(unittest.TestCase):
    def test_threshold_comes_from_separable_validation_scores(self) -> None:
        labels = [0, 0, 0, 1, 1, 1]
        scores = [0.1, 0.2, 0.15, 0.9, 0.8, 0.85]
        threshold, metrics = select_threshold(labels, scores)
        self.assertAlmostEqual(threshold, 0.5)
        self.assertEqual(metrics["f1"], 1.0)
        self.assertEqual(metrics["sensitivity"], 1.0)
        self.assertEqual(metrics["specificity"], 1.0)

    def test_covid_is_the_positive_class(self) -> None:
        # One missed COVID case: sensitivity 0.5. The Normal case is correct.
        metrics = binary_metrics([1, 1, 0], [0.2, 0.9, 0.2], threshold=0.5)
        self.assertEqual(metrics["tp"], 1)
        self.assertEqual(metrics["fn"], 1)
        self.assertEqual(metrics["tn"], 1)
        self.assertEqual(metrics["fp"], 0)
        self.assertAlmostEqual(metrics["sensitivity"], 0.5)
        self.assertAlmostEqual(metrics["specificity"], 1.0)

    def test_always_normal_baseline(self) -> None:
        baseline = always_normal_baseline([0, 0, 1, 1])
        self.assertAlmostEqual(baseline["accuracy"], 0.5)
        self.assertEqual(baseline["sensitivity"], 0.0)
        self.assertEqual(baseline["specificity"], 1.0)
        self.assertEqual(baseline["precision"], 0.0)
        self.assertEqual(baseline["f1"], 0.0)
        self.assertEqual(baseline["roc_auc"], 0.5)
        self.assertAlmostEqual(baseline["pr_auc"], 0.5)


class DiscoveryTests(unittest.TestCase):
    def test_masks_and_other_classes_are_not_model_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "COVID-19_Radiography_Dataset"
            for class_name in ("COVID", "Normal", "Lung_Opacity", "Viral Pneumonia"):
                (root / class_name / "images").mkdir(parents=True)
                (root / class_name / "masks").mkdir(parents=True)
                (root / class_name / "images" / f"{class_name}-1.png").write_bytes(b"not-read")
                (root / class_name / "masks" / f"{class_name}-1.png").write_bytes(b"mask")
            found = resolve_dataset_root(str(root.parent))
            self.assertEqual(found, root.resolve())
            covid = list_xray_images(found / "COVID")
            self.assertEqual([path.name for path in covid], ["COVID-1.png"])
            self.assertEqual(count_mask_files(found / "COVID"), 1)
            self.assertEqual(ignored_directories(found), ["Lung_Opacity", "Viral Pneumonia"])

    def test_missing_dataset_asks_for_a_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(DatasetNotFoundError):
                resolve_dataset_root(temporary)

    def test_workspace_dataset_uses_image_folders_only(self) -> None:
        dataset = PROJECT_ROOT / "COVID-19_Radiography_Dataset"
        if not dataset.is_dir():
            self.skipTest("COVID-19 Radiography Database is not in the project folder")
        root = resolve_dataset_root(str(dataset))
        covid = list_xray_images(root / "COVID")
        normal = list_xray_images(root / "Normal")
        self.assertGreater(len(covid), 0)
        self.assertGreater(len(normal), 0)
        self.assertTrue(all("masks" not in path.parts for path in covid + normal))
        self.assertGreater(count_mask_files(root / "COVID"), 0)
        ignored = set(ignored_directories(root))
        self.assertIn("Lung_Opacity", ignored)
        self.assertIn("Viral Pneumonia", ignored)


class MetadataTests(unittest.TestCase):
    def test_patient_column_is_used_and_file_name_is_not(self) -> None:
        usable = assess_patient_identifiers(
            [
                {
                    "class_name": "COVID",
                    "path": "metadata.xlsx",
                    "columns": ["FILE NAME", "patient_id"],
                    "rows": [
                        {"FILE NAME": "COVID-1", "patient_id": "p1"},
                        {"FILE NAME": "COVID-2", "patient_id": "p1"},
                    ],
                }
            ]
        )
        self.assertTrue(usable["available"])
        self.assertEqual(usable["identifier_column"], "patient_id")
        self.assertFalse(usable["filenames_used_as_patient_ids"])

        filenames = assess_patient_identifiers(
            [
                {
                    "class_name": "Normal",
                    "path": "metadata.xlsx",
                    "columns": ["FILE NAME", "URL"],
                    "rows": [
                        {"FILE NAME": "Normal-1", "URL": "https://example.test/dataset"},
                        {"FILE NAME": "Normal-2", "URL": "https://example.test/dataset"},
                    ],
                }
            ]
        )
        self.assertFalse(filenames["available"])
        self.assertIn("Patient independence cannot be verified", filenames["message"])

    def test_real_metadata_has_no_patient_identifier(self) -> None:
        dataset = PROJECT_ROOT / "COVID-19_Radiography_Dataset"
        if not (dataset / "COVID.metadata.xlsx").is_file():
            self.skipTest("metadata workbook is not in the project folder")
        tables = []
        for class_name in ("COVID", "Normal"):
            path = dataset / f"{class_name}.metadata.xlsx"
            columns, rows = read_metadata_workbook(path)
            self.assertEqual(columns, ["FILE NAME", "FORMAT", "SIZE", "URL"])
            self.assertGreater(len(rows), 0)
            tables.append({"class_name": class_name, "path": str(path), "columns": columns, "rows": rows})
        decision = assess_patient_identifiers(tables)
        self.assertFalse(decision["available"])
        self.assertFalse(decision["filenames_used_as_patient_ids"])
        self.assertIn("Patient independence cannot be verified", decision["message"])


class PrepareTests(unittest.TestCase):
    def test_synthetic_dataset_groups_duplicates_and_skips_masks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "COVID-19_Radiography_Dataset"
            output = Path(temporary) / "outputs"
            covid_dir = root / "COVID" / "images"
            normal_dir = root / "Normal" / "images"
            (root / "COVID" / "masks").mkdir(parents=True)
            (root / "Lung_Opacity" / "images").mkdir(parents=True)
            covid_dir.mkdir(parents=True)
            normal_dir.mkdir(parents=True)
            (root / "COVID" / "masks" / "COVID-1.png").write_bytes(_png_bytes((1, 2, 3)))
            (root / "Lung_Opacity" / "images" / "Lung_Opacity-1.png").write_bytes(_png_bytes((4, 5, 6)))

            normal_conflict = _png_bytes((10, 20, 30))
            for index in range(1, 21):
                (covid_dir / f"COVID-{index}.png").write_bytes(_png_bytes((index, 0, 0)))
                (normal_dir / f"Normal-{index}.png").write_bytes(
                    normal_conflict if index == 1 else _png_bytes((0, index, 0))
                )
            (covid_dir / "COVID-1-copy.png").write_bytes((covid_dir / "COVID-1.png").read_bytes())
            (covid_dir / "COVID-conflict.png").write_bytes(normal_conflict)

            report = prepare(data_dir=str(root), output_dir=output, seed=42)
            rows = load_manifest(output / "splits" / "manifest.csv")
            self.assertTrue(all("masks" not in Path(row["relative_path"]).parts for row in rows))
            self.assertTrue(all("Lung_Opacity" not in row["relative_path"] for row in rows))
            self.assertFalse(report["patient_identifiers"]["patient_independence_verified"])
            self.assertIn("Patient independence cannot be verified", report["patient_identifiers"]["message"])

            excluded = [row for row in rows if row["split"] == "excluded"]
            self.assertEqual(len(excluded), 2)
            self.assertEqual(len({row["sha256"] for row in excluded}), 1)
            self.assertEqual(sorted(row["label_name"] for row in excluded), ["COVID", "Normal"])

            copies = [row for row in rows if row["relative_path"] in {"COVID/images/COVID-1.png", "COVID/images/COVID-1-copy.png"}]
            self.assertEqual(len(copies), 2)
            self.assertEqual(copies[0]["split"], copies[1]["split"])
            self.assertNotEqual(copies[0]["split"], "excluded")

            splits_for_group: dict[str, set[str]] = {}
            for row in rows:
                splits_for_group.setdefault(row["group_id"], set()).add(row["split"])
            self.assertTrue(all(len(splits) == 1 for splits in splits_for_group.values()))

            for split in ("train", "val", "test"):
                labels = {row["label_name"] for row in rows if row["split"] == split}
                self.assertEqual(labels, {"COVID", "Normal"})

            manifest_bytes = (output / "splits" / "manifest.csv").read_bytes()
            reused = prepare(data_dir=str(root), output_dir=output, seed=42)
            self.assertTrue(reused["reused"])
            self.assertEqual((output / "splits" / "manifest.csv").read_bytes(), manifest_bytes)


def _content_box(image: Image.Image) -> tuple[int, int, int, int]:
    pixels = image.load()
    width, height = image.size
    xs = []
    ys = []
    for y in range(height):
        for x in range(width):
            if pixels[x, y][0] > 200:
                xs.append(x)
                ys.append(y)
    return min(xs), min(ys), max(xs), max(ys)


class CommandTests(unittest.TestCase):
    def test_train_command_does_not_start_by_itself(self) -> None:
        from covid_cxr.__main__ import main

        self.assertEqual(main(["train"]), 2)


if __name__ == "__main__":
    unittest.main()
