# aggregate_features.py
# Build rolling-window features.parquet from samples.csv using eICU tables (optimized).
# Apache-derived features removed to prevent indirect leakage.

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import config


#############################
# Rolling window numeric aggregator
#############################
class WindowAgg:
    def __init__(self, n_rows: int, col_names: List[str]) -> None:
        self.n = n_rows
        self.cols = col_names

        self.count = {c: np.zeros(self.n, dtype=np.int32) for c in self.cols}
        self.sum = {c: np.zeros(self.n, dtype=np.float64) for c in self.cols}
        self.sumsq = {c: np.zeros(self.n, dtype=np.float64) for c in self.cols}
        self.minv = {c: np.full(self.n, np.inf, dtype=np.float64) for c in self.cols}
        self.maxv = {c: np.full(self.n, -np.inf, dtype=np.float64) for c in self.cols}

        self.last_time = {c: np.full(self.n, -1, dtype=np.int32) for c in self.cols}
        self.last_val = {c: np.full(self.n, np.nan, dtype=np.float64) for c in self.cols}

    def update(self, row_idx: np.ndarray, t: np.ndarray, values: Dict[str, np.ndarray]) -> None:
        for c, v in values.items():
            mask = ~np.isnan(v)
            if not np.any(mask):
                continue

            idx = row_idx[mask]
            vv = v[mask]
            tt = t[mask].astype(np.int32)

            self.count[c][idx] += 1
            self.sum[c][idx] += vv
            self.sumsq[c][idx] += vv * vv
            self.minv[c][idx] = np.minimum(self.minv[c][idx], vv)
            self.maxv[c][idx] = np.maximum(self.maxv[c][idx], vv)

            lt = self.last_time[c][idx]
            newer = tt > lt
            if np.any(newer):
                idx2 = idx[newer]
                self.last_time[c][idx2] = tt[newer]
                self.last_val[c][idx2] = vv[newer]

    def finalize(self, prefix: str) -> pd.DataFrame:
        out = {}
        for c in self.cols:
            cnt = self.count[c].astype(np.float64)

            mean = np.full(self.n, np.nan, dtype=np.float64)
            std = np.full(self.n, np.nan, dtype=np.float64)
            mn = np.full(self.n, np.nan, dtype=np.float64)
            mx = np.full(self.n, np.nan, dtype=np.float64)
            last = self.last_val[c].copy()

            nonzero = cnt > 0
            if np.any(nonzero):
                mean[nonzero] = self.sum[c][nonzero] / cnt[nonzero]
                var = (self.sumsq[c][nonzero] / cnt[nonzero]) - (mean[nonzero] ** 2)
                var = np.maximum(var, 0.0)
                std[nonzero] = np.sqrt(var)
                mn[nonzero] = self.minv[c][nonzero]
                mx[nonzero] = self.maxv[c][nonzero]

            out[f"{prefix}{c}_min"] = mn
            out[f"{prefix}{c}_max"] = mx
            out[f"{prefix}{c}_mean"] = mean
            out[f"{prefix}{c}_std"] = std
            out[f"{prefix}{c}_count"] = self.count[c].astype(np.float64)
            out[f"{prefix}{c}_last"] = last

        return pd.DataFrame(out)


#############################
# Helpers
#############################
def safe_exists(path: Path) -> bool:
    return path.exists() and path.is_file()


def load_samples(samples_path: Path) -> pd.DataFrame:
    df = pd.read_csv(samples_path)
    need = {"patientunitstayid", "t_end", "label"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"samples file missing columns: {sorted(missing)}")
    df["patientunitstayid"] = df["patientunitstayid"].astype(int)
    df["t_end"] = df["t_end"].astype(int)
    df["label"] = df["label"].astype(int)
    return df


def make_sample_index_df(samples_df: pd.DataFrame) -> pd.DataFrame:
    idx_df = samples_df[["patientunitstayid", "t_end"]].copy()
    idx_df["row_idx"] = np.arange(len(idx_df), dtype=np.int32)
    return idx_df


def ceil_to_stride(t: np.ndarray, stride: int) -> np.ndarray:
    return ((t + stride - 1) // stride) * stride


#############################
# Static features (safe only)
#############################
def build_static_features(data_dir: Path, samples_df: pd.DataFrame) -> pd.DataFrame:
    pids_set = set(samples_df["patientunitstayid"].unique().tolist())
    out = pd.DataFrame(index=np.arange(len(samples_df)))

    patient_path = data_dir / "patient.csv.gz"
    if safe_exists(patient_path):
        patient = pd.read_csv(patient_path, compression="infer", low_memory=False)
        patient = patient[patient["patientunitstayid"].isin(pids_set)].copy()

        patient["age_num"] = pd.to_numeric(patient.get("age", np.nan), errors="coerce")

        static_cols = [
            "patientunitstayid",
            "age_num",
            "admissionheight",
            "admissionweight",
            "gender",
            "ethnicity",
            "unittype",
            "unitadmitsource",
            "unitstaytype",
        ]
        static_cols = [c for c in static_cols if c in patient.columns]
        patient_static = patient[static_cols].set_index("patientunitstayid")

        cat_cols = [
            c for c in
            ["gender", "ethnicity", "unittype", "unitadmitsource", "unitstaytype"]
            if c in patient_static.columns
        ]
        if cat_cols:
            patient_static = pd.get_dummies(patient_static, columns=cat_cols, dummy_na=True)

        merged = samples_df[["patientunitstayid"]].merge(
            patient_static.reset_index(),
            on="patientunitstayid",
            how="left",
        ).drop(columns=["patientunitstayid"])
        merged.columns = [f"pt_{c}" for c in merged.columns]
        out = pd.concat([out, merged], axis=1)
    else:
        print(f"WARNING: missing file, skipping patient static: {patient_path}")

    return out
