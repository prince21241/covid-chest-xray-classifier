"""Find COVID and Normal X-ray folders without using mask images."""

from __future__ import annotations

import os
from pathlib import Path

from covid_cxr.config import CLASS_DIRECTORY_NAMES, PROJECT_ROOT

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
MASK_DIR_NAMES = {"mask", "masks"}


class DatasetNotFoundError(FileNotFoundError):
    """Raised when COVID and Normal image folders cannot be located."""


def find_named_child(root: Path, name: str) -> Path | None:
    if not root.is_dir():
        return None
    for child in root.iterdir():
        if child.is_dir() and child.name.lower() == name.lower():
            return child
    return None


def has_both_classes(root: Path) -> bool:
    return all(find_named_child(root, name) is not None for name in CLASS_DIRECTORY_NAMES)


def is_mask_path(path: Path, class_dir: Path) -> bool:
    relative = path.relative_to(class_dir)
    return any(part.lower() in MASK_DIR_NAMES for part in relative.parts)


def list_xray_images(class_dir: Path) -> list[Path]:
    """Return X-ray files for one class.

    When an ``images`` directory exists, only that directory is searched.
    Paths inside a ``mask`` or ``masks`` directory are never returned.
    """
    images_dir = find_named_child(class_dir, "images")
    search_root = images_dir if images_dir is not None else class_dir
    found: list[Path] = []
    for path in search_root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        if is_mask_path(path, class_dir):
            continue
        found.append(path)
    return sorted(found)


def count_mask_files(class_dir: Path) -> int:
    total = 0
    if not class_dir.is_dir():
        return 0
    for path in class_dir.rglob("*"):
        if path.is_file() and is_mask_path(path, class_dir):
            total += 1
    return total


def ignored_directories(root: Path) -> list[str]:
    used = {name.lower() for name in CLASS_DIRECTORY_NAMES}
    names: list[str] = []
    for child in root.iterdir():
        if child.is_dir() and child.name.lower() not in used:
            names.append(child.name)
    return sorted(names)


def _candidate_roots(start: Path) -> list[Path]:
    if not start.exists() or not start.is_dir():
        return []
    candidates = [start]
    children = [child for child in start.iterdir() if child.is_dir()]
    if len(children) > 40:
        children = [
            child
            for child in children
            if "covid" in child.name.lower() or "radiography" in child.name.lower()
        ]
    for child in children:
        candidates.append(child)
        grandchildren = [item for item in child.iterdir() if item.is_dir()]
        if len(grandchildren) > 40:
            grandchildren = [
                item
                for item in grandchildren
                if "covid" in item.name.lower() or "radiography" in item.name.lower()
            ]
        candidates.extend(grandchildren)
    return candidates


def _image_count(root: Path) -> int:
    total = 0
    for class_name in CLASS_DIRECTORY_NAMES:
        class_dir = find_named_child(root, class_name)
        if class_dir is not None:
            total += len(list_xray_images(class_dir))
    return total


def locate_dataset_root(start: Path) -> Path | None:
    matches = [candidate for candidate in _candidate_roots(start) if has_both_classes(candidate)]
    if not matches:
        return None
    unique: list[Path] = []
    seen: set[Path] = set()
    for match in matches:
        resolved = match.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(resolved)
    return max(unique, key=_image_count)


def resolve_dataset_root(data_dir: str | None) -> Path:
    """Find the folder that directly contains COVID and Normal.

    An explicit ``data_dir`` is required to contain the dataset. Automatic
    search is used only when no path was given.
    """
    if data_dir:
        start = Path(data_dir).expanduser()
        if not start.exists():
            raise DatasetNotFoundError(
                f"Dataset path does not exist: {start}\n"
                "Pass the folder that contains the COVID and Normal directories."
            )
        located = locate_dataset_root(start)
        if located is None:
            seen = []
            if start.is_dir():
                seen = sorted(child.name for child in start.iterdir() if child.is_dir())
            raise DatasetNotFoundError(
                "Could not find both COVID and Normal image folders under:\n"
                f"  {start.resolve()}\n"
                f"Directories seen: {seen or '(none)'}\n"
                "Point --data-dir at the folder that contains COVID and Normal. "
                "Mask folders are ignored, and other categories are not used."
            )
        return located

    searched: list[Path] = []
    env_dir = os.environ.get("CXR_DATA_DIR")
    starts: list[Path] = []
    if env_dir:
        starts.append(Path(env_dir).expanduser())
    starts.extend(
        [
            Path.cwd(),
            Path.cwd() / "COVID-19_Radiography_Dataset",
            PROJECT_ROOT,
            PROJECT_ROOT / "COVID-19_Radiography_Dataset",
        ]
    )
    for start in starts:
        searched.append(start)
        if not start.exists():
            continue
        located = locate_dataset_root(start)
        if located is not None:
            return located

    searched_text = "\n".join(f"  - {path}" for path in searched)
    raise DatasetNotFoundError(
        "Could not find the COVID and Normal image folders.\n"
        f"Searched:\n{searched_text}\n"
        "Pass the dataset folder explicitly, for example:\n"
        '  python -m covid_cxr prepare --data-dir "C:\\path\\to\\COVID-19_Radiography_Dataset"'
    )
