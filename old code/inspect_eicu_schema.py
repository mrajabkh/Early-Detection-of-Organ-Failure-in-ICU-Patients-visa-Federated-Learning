# inspect_eicu_schema.py
# Dump eICU folder file list + schema (columns, offset candidates) to a report.
# Location: Project/Code/inspect_eicu_schema.py

from __future__ import annotations

import gzip
import io
import os
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

import config


#############################
# Helpers
#############################
def is_csv_like(p: Path) -> bool:
    name = p.name.lower()
    return name.endswith(".csv.gz") or name.endswith(".csv")


def read_header_csv_gz(path: Path, nrows: int = 3) -> pd.DataFrame:
    # pandas can read .csv.gz directly, but this avoids some edge weirdness
    return pd.read_csv(path, compression="infer", nrows=nrows, low_memory=False)


def candidate_offset_columns(cols: List[str]) -> List[str]:
    out = []
    for c in cols:
        cl = c.lower()
        if "offset" in cl or cl.endswith("time") or cl.endswith("timestamp"):
            out.append(c)
    return out


def summarize_file(path: Path) -> str:
    lines: List[str] = []
    lines.append(f"FILE: {path.name}")

    try:
        df = read_header_csv_gz(path, nrows=3)
    except Exception as e:
        lines.append(f"  ERROR reading header/sample: {repr(e)}")
        return "\n".join(lines)

    cols = df.columns.tolist()
    lines.append(f"  Columns ({len(cols)}): {cols}")

    offsets = candidate_offset_columns(cols)
    lines.append(f"  Offset/time candidates ({len(offsets)}): {offsets}")

    # Dtypes from the tiny sample (not perfect but useful)
    dtypes = {c: str(df[c].dtype) for c in cols}
    lines.append(f"  Sample dtypes: {dtypes}")

    # Numeric-like columns from sample
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    lines.append(f"  Numeric cols in sample ({len(numeric_cols)}): {numeric_cols[:25]}" + (" ..." if len(numeric_cols) > 25 else ""))

    # Small sample rows
    lines.append("  Sample rows (first 3):")
    lines.append(df.head(3).to_string(index=False))

    return "\n".join(lines)


#############################
# Main
#############################
def main() -> None:
    data_dir = config.EICU_DATA_DIR
    if not data_dir.exists():
        raise FileNotFoundError(f"Could not find eICU folder at: {data_dir}")

    out_dir = config.run_dir(config.DISEASE)
    report_path = out_dir / f"eicu_schema_report__{config.disease_tag(config.DISEASE)}.txt"

    all_paths = sorted([p for p in data_dir.iterdir() if p.is_file()])
    csv_paths = [p for p in all_paths if is_csv_like(p)]

    lines: List[str] = []
    lines.append("#############################")
    lines.append("eICU SCHEMA REPORT")
    lines.append("#############################")
    lines.append(f"Data dir: {data_dir}")
    lines.append(f"Total files: {len(all_paths)}")
    lines.append(f"CSV-like files: {len(csv_paths)}")
    lines.append("")
    lines.append("FILES:")
    for p in all_paths:
        lines.append(f"  - {p.name}")
    lines.append("")

    for p in csv_paths:
        lines.append("#############################")
        lines.append(summarize_file(p))
        lines.append("")

    report_text = "\n".join(lines)

    # Print to console (truncated if huge)
    print(report_text[:20000])
    if len(report_text) > 20000:
        print("\n... (output truncated in console) ...\n")

    # Save full report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    print("#############################")
    print(f"Saved full report to: {report_path}")
    print("#############################")


if __name__ == "__main__":
    main()
