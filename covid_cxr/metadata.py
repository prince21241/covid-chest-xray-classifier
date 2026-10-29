"""Read the dataset's metadata workbooks and decide whether a patient ID exists.

Filenames are never treated as patient identifiers. A source URL shared by many
images is a dataset link, not a patient ID.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
import zipfile
from collections import Counter
from pathlib import Path

SPREADSHEET_NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
REL_NS = {"r": "http://schemas.openxmlformats.org/package/2006/relationships"}
OFFICE_REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"

NON_IDENTIFIER_NAMES = {
    "file name",
    "filename",
    "file_name",
    "file",
    "format",
    "size",
    "url",
    "image",
    "path",
    "mask",
    "label",
    "class",
    "source",
    "doi",
}
IDENTIFIER_HINTS = ("patient", "subject", "person_id", "person id", "pid")


def _column_index(letters: str) -> int:
    index = 0
    for char in letters:
        if char.isalpha():
            index = index * 26 + (ord(char.upper()) - 64)
    return index


def _cell_value(cell: ET.Element, shared_strings: list[str]) -> str:
    cell_type = cell.get("t")
    if cell_type == "inlineStr":
        inline = cell.find("m:is", SPREADSHEET_NS)
        if inline is None:
            return ""
        return "".join(node.text or "" for node in inline.findall(".//m:t", SPREADSHEET_NS)).strip()
    value = cell.find("m:v", SPREADSHEET_NS)
    if value is None or value.text is None:
        return ""
    if cell_type == "s":
        return shared_strings[int(value.text)].strip()
    return value.text.strip()


def _sheet_path(workbook: zipfile.ZipFile) -> str:
    if "xl/workbook.xml" not in workbook.namelist():
        return "xl/worksheets/sheet1.xml"
    root = ET.fromstring(workbook.read("xl/workbook.xml"))
    sheet = root.find("m:sheets/m:sheet", SPREADSHEET_NS)
    if sheet is None:
        return "xl/worksheets/sheet1.xml"
    relation_id = sheet.get(f"{{{OFFICE_REL}}}id")
    rels_name = "xl/_rels/workbook.xml.rels"
    if not relation_id or rels_name not in workbook.namelist():
        return "xl/worksheets/sheet1.xml"
    rels = ET.fromstring(workbook.read(rels_name))
    for rel in rels:
        if rel.get("Id") == relation_id:
            target = rel.get("Target", "worksheets/sheet1.xml").lstrip("/")
            if target.startswith("xl/"):
                return target
            return "xl/" + target
    return "xl/worksheets/sheet1.xml"


def read_metadata_workbook(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    """Read the first worksheet of a simple metadata .xlsx file."""
    with zipfile.ZipFile(path) as workbook:
        shared_strings: list[str] = []
        if "xl/sharedStrings.xml" in workbook.namelist():
            root = ET.fromstring(workbook.read("xl/sharedStrings.xml"))
            for item in root.findall("m:si", SPREADSHEET_NS):
                shared_strings.append(
                    "".join(node.text or "" for node in item.findall(".//m:t", SPREADSHEET_NS))
                )
        sheet = ET.fromstring(workbook.read(_sheet_path(workbook)))

    parsed_rows: list[dict[int, str]] = []
    for row in sheet.findall("m:sheetData/m:row", SPREADSHEET_NS):
        values: dict[int, str] = {}
        for cell in row.findall("m:c", SPREADSHEET_NS):
            ref = cell.get("r") or ""
            letters = "".join(char for char in ref if char.isalpha())
            values[_column_index(letters)] = _cell_value(cell, shared_strings)
        if any(value.strip() for value in values.values()):
            parsed_rows.append(values)
    if not parsed_rows:
        return [], []

    header_cells = parsed_rows[0]
    columns = sorted(header_cells)
    header = [header_cells.get(column, "").strip() for column in columns]
    records: list[dict[str, str]] = []
    for cells in parsed_rows[1:]:
        record = {header[index]: cells.get(column, "").strip() for index, column in enumerate(columns)}
        if any(record.values()):
            records.append(record)
    return header, records


def find_metadata_files(dataset_root: Path) -> dict[str, Path]:
    """Map class name to its metadata workbook, when one is present."""
    found: dict[str, Path] = {}
    for class_name in ("COVID", "Normal"):
        candidates = [
            dataset_root / f"{class_name}.metadata.xlsx",
            dataset_root / class_name / f"{class_name}.metadata.xlsx",
        ]
        for candidate in candidates:
            if candidate.is_file():
                found[class_name] = candidate
                break
        if class_name in found:
            continue
        class_dir = dataset_root / class_name
        search_roots = [dataset_root]
        if class_dir.is_dir():
            search_roots.append(class_dir)
        for folder in search_roots:
            for path in folder.glob("*.xlsx"):
                if path.name.lower().startswith(class_name.lower()) and "metadata" in path.name.lower():
                    found[class_name] = path
                    break
            if class_name in found:
                break
    return found


def _classify_column(name: str, values: list[str]) -> dict:
    normalized = " ".join(name.strip().lower().replace("_", " ").split())
    nonempty = [value for value in values if value.strip()]
    unique = len(set(nonempty))
    info = {
        "name": name,
        "usable": False,
        "reason": "",
        "n_rows": len(values),
        "n_nonempty": len(nonempty),
        "n_unique": unique,
        "common_values": [],
    }
    if unique <= 12 and nonempty:
        info["common_values"] = [
            {"value": value, "count": count} for value, count in Counter(nonempty).most_common(12)
        ]

    if not normalized:
        info["reason"] = "Empty column name."
        return info
    if normalized in NON_IDENTIFIER_NAMES or normalized.replace(" ", "") in {"filename", "filepath"}:
        if normalized == "url":
            info["reason"] = (
                "Source link, not a patient identifier "
                f"({unique} distinct values across {len(values)} rows)."
            )
        elif normalized in {"file name", "filename", "file_name", "file"}:
            info["reason"] = "File name, not a patient identifier. Filenames were not parsed as patient IDs."
        else:
            info["reason"] = "File metadata, not a patient identifier."
        return info
    if not any(hint in normalized for hint in IDENTIFIER_HINTS):
        info["reason"] = "Column name is not a patient or subject identifier."
        return info
    if len(values) == 0 or len(nonempty) < 0.95 * len(values):
        info["reason"] = "Too many empty values to use this column as a patient identifier."
        return info
    info["usable"] = True
    info["reason"] = "Column name describes a patient or subject identifier."
    return info


def assess_patient_identifiers(tables: list[dict]) -> dict:
    """Decide whether metadata can support a patient-level split.

    ``tables`` items contain ``class_name``, ``path``, ``columns``, and ``rows``.
    The result never invents an identifier from the file name.
    """
    column_reports: list[dict] = []
    usable_by_key: dict[str, list[str]] = {}
    if not tables:
        return {
            "available": False,
            "patient_independence_verified": False,
            "filenames_used_as_patient_ids": False,
            "identifier_column": None,
            "columns": [],
            "message": (
                "Patient independence cannot be verified. No metadata workbook was found. "
                "Filenames were not treated as patient IDs."
            ),
        }

    for table in tables:
        rows = table["rows"]
        for column in table["columns"]:
            values = [str(row.get(column, "")).strip() for row in rows]
            report = _classify_column(column, values)
            report["class_name"] = table["class_name"]
            report["metadata_file"] = str(table["path"])
            column_reports.append(report)
            if report["usable"]:
                usable_by_key.setdefault(column, values)

    usable_names = sorted(usable_by_key)
    identifier = None
    available = False
    if len(usable_names) == 1:
        identifier = usable_names[0]
        available = True
        message = (
            f"Using metadata column {identifier!r} as the patient identifier. "
            "Images from the same identifier will stay in one split."
        )
    elif len(usable_names) > 1:
        message = (
            "Patient independence cannot be verified. More than one column looked like a "
            f"patient identifier ({', '.join(usable_names)}), so none was used. "
            "Filenames were not treated as patient IDs."
        )
    else:
        names = []
        for table in tables:
            for column in table["columns"]:
                label = f"{table['class_name']}: {column}"
                if label not in names:
                    names.append(label)
        shown = ", ".join(names) if names else "(none)"
        message = (
            "Patient independence cannot be verified. "
            f"Metadata columns found: {shown}. "
            "No column is a reliable patient identifier. "
            "Filenames were not treated as patient IDs. URL values, when present, are source links. "
            "Exact duplicate files are kept in the same split, but different images of the same "
            "patient can still fall into different splits."
        )

    return {
        "available": available,
        "patient_independence_verified": available,
        "filenames_used_as_patient_ids": False,
        "identifier_column": identifier,
        "columns": column_reports,
        "message": message,
    }


def patient_id_lookup(tables: list[dict], identifier_column: str) -> dict[tuple[str, str], str]:
    """Map ``(class name lower, file stem lower)`` to a patient ID."""
    lookup: dict[tuple[str, str], str] = {}
    for table in tables:
        class_name = table["class_name"].lower()
        for row in table["rows"]:
            file_name = ""
            for key, value in row.items():
                if key.strip().lower() in {"file name", "filename", "file_name"}:
                    file_name = str(value).strip()
                    break
            patient_id = str(row.get(identifier_column, "")).strip()
            if file_name and patient_id:
                stem = Path(file_name).stem.lower()
                lookup[(class_name, stem)] = patient_id
    return lookup
