"""Group images so duplicates (and patients, when known) stay in one split."""

from __future__ import annotations

import random
from collections import defaultdict

from covid_cxr.config import TEST_RATIO, TRAIN_RATIO, VAL_RATIO

SPLIT_NAMES = ("train", "val", "test")
RATIOS = {"train": TRAIN_RATIO, "val": VAL_RATIO, "test": TEST_RATIO}


class UnionFind:
    def __init__(self, items: list[str]) -> None:
        self.parent = {item: item for item in items}

    def find(self, item: str) -> str:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def group_records(records: list[dict], use_patient_ids: bool) -> list[dict]:
    """Connect records that must not be separated.

    Exact file hashes are always connected. Patient IDs are connected only when
    metadata actually provided them. A group with more than one label is a
    conflict and must not be trained on.
    """
    if not records:
        return []
    nodes = [record["relative_path"] for record in records]
    if len(nodes) != len(set(nodes)):
        raise ValueError("Each image path must be unique.")
    union = UnionFind(nodes)
    by_hash: dict[str, list[str]] = defaultdict(list)
    by_patient: dict[str, list[str]] = defaultdict(list)
    for record in records:
        by_hash[record["sha256"]].append(record["relative_path"])
        patient_id = (record.get("patient_id") or "").strip()
        if use_patient_ids and patient_id:
            by_patient[patient_id].append(record["relative_path"])
    for paths in list(by_hash.values()) + list(by_patient.values()):
        first = paths[0]
        for other in paths[1:]:
            union.union(first, other)

    members: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        members[union.find(record["relative_path"])].append(record)

    groups: list[dict] = []
    for grouped in members.values():
        labels = {int(record["label"]) for record in grouped}
        hashes = {record["sha256"] for record in grouped}
        anchor = min(record["relative_path"] for record in grouped)
        if len(hashes) == 1:
            group_id = next(iter(hashes))
        else:
            group_id = "component:" + anchor
        groups.append(
            {
                "group_id": group_id,
                "records": grouped,
                "label": next(iter(labels)) if len(labels) == 1 else None,
                "conflict": len(labels) > 1,
                "n": len(grouped),
                "paths": [record["relative_path"] for record in grouped],
            }
        )
    return groups


def assign_splits(groups: list[dict], seed: int) -> dict[str, str]:
    """Assign each non-conflicting group to one split.

    Within each label, groups are shuffled and then filled in order: training
    until it reaches about 70% of that label's images, then validation to about
    15%, and the rest to test. A group that crosses a boundary stays entirely in
    the split being filled.
    """
    by_label: dict[int, list[dict]] = defaultdict(list)
    for group in groups:
        if group["conflict"] or group["label"] is None:
            continue
        by_label[int(group["label"])].append(group)

    assignment: dict[str, str] = {}
    rng = random.Random(seed)
    for label in sorted(by_label):
        items = sorted(by_label[label], key=lambda group: group["group_id"])
        rng.shuffle(items)
        total = sum(int(group["n"]) for group in items)
        targets = [RATIOS[name] * total for name in SPLIT_NAMES]
        counts = [0, 0, 0]
        split_index = 0
        for group in items:
            while split_index < 2 and counts[split_index] >= targets[split_index]:
                split_index += 1
            assignment[group["group_id"]] = SPLIT_NAMES[split_index]
            counts[split_index] += int(group["n"])
    return assignment
