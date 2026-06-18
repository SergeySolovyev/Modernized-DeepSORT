"""Versioned experiment journal (results/experiments.csv).

Every eval run appends a row here so the full experimental history is traceable
(a webinar requirement). Rows may have heterogeneous columns (detector P/R/F1,
HOTA, FPS); the header is expanded and the file rewritten when new keys appear.
"""
import csv
import os

DEFAULT_JOURNAL = os.path.join("results", "experiments.csv")


def append_row(row, csv_path=DEFAULT_JOURNAL):
    """Append a dict row, expanding the CSV header if it introduces new keys."""
    csv_path = csv_path or DEFAULT_JOURNAL
    parent = os.path.dirname(os.path.abspath(csv_path))
    os.makedirs(parent, exist_ok=True)

    header, rows = [], []
    if os.path.exists(csv_path):
        with open(csv_path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            header = list(reader.fieldnames or [])
            rows = list(reader)

    for key in row:
        if key not in header:
            header.append(str(key))
    rows.append({k: row.get(k, "") for k in row})

    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=header)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in header})
    return csv_path
