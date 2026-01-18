# aggregate_features.py
# Build rolling-window features.parquet from samples.csv using eICU tables (optimized).
# Includes:
# - respiratoryCharting pivot (top-N numeric labels)
# - nurseCharting pivot (top-N numeric labels)
# Location: Project/Code/aggregate_features.py

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


def build_event_to_window_mapping(
    pids: np.ndarray,
    times: np.ndarray,
    sample_index_df: pd.DataFrame,
    history_mins: int,
    stride_mins: int,
) -> pd.DataFrame:
    pids = pids.astype(np.int64)
    times = times.astype(np.int64)

    t0 = ceil_to_stride(times, stride_mins)

    offs = np.arange(0, history_mins, stride_mins, dtype=np.int64)
    k = len(offs)

    pid_rep = np.repeat(pids, k)
    time_rep = np.repeat(times, k)
    event_i_rep = np.repeat(np.arange(len(times), dtype=np.int32), k)
    t_end_rep = np.repeat(t0, k) + np.tile(offs, len(times))

    in_window = (t_end_rep - history_mins < time_rep) & (time_rep <= t_end_rep)
    if not np.any(in_window):
        return pd.DataFrame(columns=["event_i", "row_idx", "time"])

    map_df = pd.DataFrame({
        "patientunitstayid": pid_rep[in_window].astype(np.int64),
        "t_end": t_end_rep[in_window].astype(np.int64),
        "event_i": event_i_rep[in_window].astype(np.int32),
        "time": time_rep[in_window].astype(np.int64),
    })

    merged = map_df.merge(
        sample_index_df,
        on=["patientunitstayid", "t_end"],
        how="inner",
    )

    return merged[["event_i", "row_idx", "time"]]


def numeric_cols_excluding_offsets(df: pd.DataFrame, pid_col: str, offset_col: str) -> List[str]:
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    out = []
    for c in num_cols:
        cl = c.lower()
        if c == pid_col:
            continue
        if c == offset_col:
            continue
        if "offset" in cl:
            continue
        out.append(c)
    return out


def safe_label_to_col(label: str) -> str:
    s = str(label).strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    if not s:
        s = "unknown"
    return s


#############################
# Static features (broadcast to windows)
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
            c for c in ["gender", "ethnicity", "unittype", "unitadmitsource", "unitstaytype"]
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

    aps_path = data_dir / "apacheApsVar.csv.gz"
    if safe_exists(aps_path):
        aps = pd.read_csv(aps_path, compression="infer", low_memory=False)
        if "patientunitstayid" in aps.columns:
            aps = aps[aps["patientunitstayid"].isin(pids_set)].copy()
            aps = aps.drop(columns=["apacheapsvarid"], errors="ignore")
            aps = aps.set_index("patientunitstayid")
            aps.columns = [f"aps_{c}" for c in aps.columns]
            merged = samples_df[["patientunitstayid"]].merge(
                aps.reset_index(),
                on="patientunitstayid",
                how="left",
            ).drop(columns=["patientunitstayid"])
            out = pd.concat([out, merged], axis=1)
    else:
        print(f"WARNING: missing file, skipping apacheApsVar: {aps_path}")

    apred_path = data_dir / "apachePredVar.csv.gz"
    if safe_exists(apred_path):
        apred = pd.read_csv(apred_path, compression="infer", low_memory=False)
        if "patientunitstayid" in apred.columns:
            apred = apred[apred["patientunitstayid"].isin(pids_set)].copy()
            apred = apred.drop(columns=["apachepredvarid"], errors="ignore")
            apred = apred.set_index("patientunitstayid")
            apred.columns = [f"apred_{c}" for c in apred.columns]
            merged = samples_df[["patientunitstayid"]].merge(
                apred.reset_index(),
                on="patientunitstayid",
                how="left",
            ).drop(columns=["patientunitstayid"])
            out = pd.concat([out, merged], axis=1)
    else:
        print(f"WARNING: missing file, skipping apachePredVar: {apred_path}")

    ares_path = data_dir / "apachePatientResult.csv.gz"
    if safe_exists(ares_path):
        ares = pd.read_csv(ares_path, compression="infer", low_memory=False)
        if "patientunitstayid" in ares.columns:
            ares = ares[ares["patientunitstayid"].isin(pids_set)].copy()
            ares = ares.drop(columns=["apachepatientresultsid"], errors="ignore")

            early_cols = [
                "acutephysiologyscore",
                "apachescore",
                "apacheversion",
                "predictedicumortality",
                "predictedhospitalmortality",
            ]
            early_cols = [c for c in early_cols if c in ares.columns]
            ares = ares[["patientunitstayid"] + early_cols].set_index("patientunitstayid")
            ares.columns = [f"ares_{c}" for c in ares.columns]

            merged = samples_df[["patientunitstayid"]].merge(
                ares.reset_index(),
                on="patientunitstayid",
                how="left",
            ).drop(columns=["patientunitstayid"])
            out = pd.concat([out, merged], axis=1)
    else:
        print(f"WARNING: missing file, skipping apachePatientResult: {ares_path}")

    return out


#############################
# Numeric time series table aggregation (optimized)
#############################
def process_numeric_timeseries_table(
    table_path: Path,
    pid_col: str,
    offset_col: str,
    usecols: List[str],
    drop_cols: Optional[List[str]],
    prefix: str,
    sample_index_df: pd.DataFrame,
    n_samples: int,
    history_mins: int,
    stride_mins: int,
    chunksize: int = 1_000_000,
) -> pd.DataFrame:
    print("#############################")
    print(f"Processing table: {table_path.name}")
    print(f"Offset col: {offset_col}")
    print(f"Prefix: {prefix}")
    print("#############################")

    if not safe_exists(table_path):
        print(f"WARNING: missing file, skipping: {table_path}")
        return pd.DataFrame(index=np.arange(n_samples))

    if drop_cols is None:
        drop_cols = []

    cols = list(dict.fromkeys([pid_col, offset_col] + usecols))
    reader = pd.read_csv(
        table_path,
        compression="infer",
        low_memory=False,
        chunksize=chunksize,
        usecols=cols,
    )

    agg: Optional[WindowAgg] = None
    numeric_cols: Optional[List[str]] = None
    mapped_total = 0

    for chunk in reader:
        if pid_col not in chunk.columns or offset_col not in chunk.columns:
            continue

        chunk = chunk.drop(columns=drop_cols, errors="ignore")

        chunk[pid_col] = pd.to_numeric(chunk[pid_col], errors="coerce")
        chunk[offset_col] = pd.to_numeric(chunk[offset_col], errors="coerce")
        chunk = chunk.dropna(subset=[pid_col, offset_col])
        if chunk.empty:
            continue

        chunk[pid_col] = chunk[pid_col].astype(np.int64)
        chunk[offset_col] = chunk[offset_col].astype(np.int64)

        for c in chunk.columns:
            if c in [pid_col, offset_col]:
                continue
            chunk[c] = pd.to_numeric(chunk[c], errors="coerce")

        if numeric_cols is None:
            numeric_cols = numeric_cols_excluding_offsets(chunk, pid_col, offset_col)
            if not numeric_cols:
                print("No numeric signal columns found (after offset filtering). Skipping.")
                return pd.DataFrame(index=np.arange(n_samples))
            agg = WindowAgg(n_rows=n_samples, col_names=numeric_cols)

        assert agg is not None
        assert numeric_cols is not None

        pids = chunk[pid_col].to_numpy(dtype=np.int64)
        times = chunk[offset_col].to_numpy(dtype=np.int64)

        map_df = build_event_to_window_mapping(
            pids=pids,
            times=times,
            sample_index_df=sample_index_df,
            history_mins=history_mins,
            stride_mins=stride_mins,
        )
        if map_df.empty:
            continue

        mapped_total += len(map_df)

        event_i = map_df["event_i"].to_numpy(dtype=np.int32)
        row_idx = map_df["row_idx"].to_numpy(dtype=np.int32)
        t = map_df["time"].to_numpy(dtype=np.int64)

        values = {}
        for c in numeric_cols:
            arr = chunk[c].to_numpy(dtype=np.float64)
            values[c] = arr[event_i]

        agg.update(row_idx=row_idx, t=t, values=values)

    if agg is None:
        return pd.DataFrame(index=np.arange(n_samples))

    print(f"Mapped event->window pairs: {mapped_total}")
    return agg.finalize(prefix=prefix)


#############################
# Charting pivots (top-N numeric labels)
#############################
def find_top_numeric_labels(
    table_path: Path,
    label_col: str,
    value_col: str,
    top_n: int,
    chunksize: int = 800_000,
) -> List[str]:
    if not safe_exists(table_path):
        return []

    counts: Dict[str, int] = {}

    reader = pd.read_csv(
        table_path,
        compression="infer",
        low_memory=False,
        chunksize=chunksize,
        usecols=[label_col, value_col],
    )

    for chunk in reader:
        vals = pd.to_numeric(chunk[value_col], errors="coerce")
        ok = vals.notna()
        if not ok.any():
            continue
        labels = chunk.loc[ok, label_col].astype(str)
        vc = labels.value_counts()
        for k, v in vc.items():
            counts[k] = counts.get(k, 0) + int(v)

    if not counts:
        return []

    sorted_labels = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    return [lbl for lbl, _ in sorted_labels[:top_n]]


def process_charting_pivot(
    table_path: Path,
    pid_col: str,
    offset_col: str,
    label_col: str,
    value_col: str,
    prefix: str,
    sample_index_df: pd.DataFrame,
    n_samples: int,
    history_mins: int,
    stride_mins: int,
    top_labels: List[str],
    chunksize: int = 800_000,
) -> pd.DataFrame:
    print("#############################")
    print(f"Processing {table_path.name} (pivot by label)")
    print(f"Top labels: {len(top_labels)}")
    print("#############################")

    if not top_labels or not safe_exists(table_path):
        return pd.DataFrame(index=np.arange(n_samples))

    label_to_agg: Dict[str, WindowAgg] = {}
    label_to_slug: Dict[str, str] = {}
    used_slugs: Dict[str, int] = {}

    for lbl in top_labels:
        base = safe_label_to_col(lbl)
        if base not in used_slugs:
            used_slugs[base] = 1
            slug = base
        else:
            used_slugs[base] += 1
            slug = f"{base}_{used_slugs[base]}"

        label_to_slug[lbl] = slug
        label_to_agg[lbl] = WindowAgg(n_rows=n_samples, col_names=["value"])

    reader = pd.read_csv(
        table_path,
        compression="infer",
        low_memory=False,
        chunksize=chunksize,
        usecols=[pid_col, offset_col, label_col, value_col],
    )

    top_set = set(top_labels)
    mapped_total = 0

    for chunk in reader:
        chunk[label_col] = chunk[label_col].astype(str)
        chunk = chunk[chunk[label_col].isin(top_set)]
        if chunk.empty:
            continue

        chunk[pid_col] = pd.to_numeric(chunk[pid_col], errors="coerce")
        chunk[offset_col] = pd.to_numeric(chunk[offset_col], errors="coerce")
        chunk["__value_num__"] = pd.to_numeric(chunk[value_col], errors="coerce")

        chunk = chunk.dropna(subset=[pid_col, offset_col, "__value_num__"])
        if chunk.empty:
            continue

        chunk[pid_col] = chunk[pid_col].astype(np.int64)
        chunk[offset_col] = chunk[offset_col].astype(np.int64)
        chunk["__value_num__"] = chunk["__value_num__"].astype(np.float64)

        for lbl, sub in chunk.groupby(label_col, sort=False):
            pids = sub[pid_col].to_numpy(dtype=np.int64)
            times = sub[offset_col].to_numpy(dtype=np.int64)

            map_df = build_event_to_window_mapping(
                pids=pids,
                times=times,
                sample_index_df=sample_index_df,
                history_mins=history_mins,
                stride_mins=stride_mins,
            )
            if map_df.empty:
                continue

            mapped_total += len(map_df)

            event_i = map_df["event_i"].to_numpy(dtype=np.int32)
            row_idx = map_df["row_idx"].to_numpy(dtype=np.int32)
            t = map_df["time"].to_numpy(dtype=np.int64)

            vals = sub["__value_num__"].to_numpy(dtype=np.float64)
            values = {"value": vals[event_i]}

            label_to_agg[lbl].update(row_idx=row_idx, t=t, values=values)

    print(f"{table_path.name} mapped event->window pairs: {mapped_total}")

    out_parts = []
    for lbl in top_labels:
        slug = label_to_slug[lbl]
        df_lbl = label_to_agg[lbl].finalize(prefix=f"{prefix}{slug}_")
        df_lbl = df_lbl.rename(columns=lambda c: c.replace(f"{prefix}{slug}_value_", f"{prefix}{slug}_"))
        out_parts.append(df_lbl)

    out = pd.concat(out_parts, axis=1) if out_parts else pd.DataFrame(index=np.arange(n_samples))

    if out.columns.duplicated().any():
        new_cols = []
        seen: Dict[str, int] = {}
        for c in out.columns:
            if c not in seen:
                seen[c] = 1
                new_cols.append(c)
            else:
                seen[c] += 1
                new_cols.append(f"{c}__dup{seen[c]}")
        out.columns = new_cols

    return out


#############################
# Count/flag features
#############################
def process_count_table(
    table_path: Path,
    pid_col: str,
    offset_col: str,
    sample_index_df: pd.DataFrame,
    n_samples: int,
    history_mins: int,
    stride_mins: int,
    chunksize: int,
    out_col: str,
) -> pd.DataFrame:
    if not safe_exists(table_path):
        print(f"WARNING: missing file, skipping: {table_path}")
        return pd.DataFrame({out_col: np.zeros(n_samples, dtype=np.float64)})

    cnt = np.zeros(n_samples, dtype=np.int32)

    reader = pd.read_csv(
        table_path,
        compression="infer",
        low_memory=False,
        chunksize=chunksize,
        usecols=[pid_col, offset_col],
    )

    mapped_total = 0
    for chunk in reader:
        chunk[pid_col] = pd.to_numeric(chunk[pid_col], errors="coerce")
        chunk[offset_col] = pd.to_numeric(chunk[offset_col], errors="coerce")
        chunk = chunk.dropna(subset=[pid_col, offset_col])
        if chunk.empty:
            continue

        pids = chunk[pid_col].astype(np.int64).to_numpy()
        times = chunk[offset_col].astype(np.int64).to_numpy()

        map_df = build_event_to_window_mapping(
            pids=pids,
            times=times,
            sample_index_df=sample_index_df,
            history_mins=history_mins,
            stride_mins=stride_mins,
        )
        if map_df.empty:
            continue

        mapped_total += len(map_df)
        row_idx = map_df["row_idx"].to_numpy(dtype=np.int32)
        np.add.at(cnt, row_idx, 1)

    print(f"{out_col}: mapped pairs {mapped_total}")
    return pd.DataFrame({out_col: cnt.astype(np.float64)})


def process_treatment_dialysis(
    table_path: Path,
    sample_index_df: pd.DataFrame,
    n_samples: int,
    history_mins: int,
    stride_mins: int,
    chunksize: int,
) -> pd.DataFrame:
    if not safe_exists(table_path):
        print(f"WARNING: missing file, skipping: {table_path}")
        return pd.DataFrame({
            "treatment_count_in_window": np.zeros(n_samples, dtype=np.float64),
            "treatment_dialysis_any": np.zeros(n_samples, dtype=np.float64),
        })

    tr_cnt = np.zeros(n_samples, dtype=np.int32)
    dial_any = np.zeros(n_samples, dtype=np.int32)

    reader = pd.read_csv(
        table_path,
        compression="infer",
        low_memory=False,
        chunksize=chunksize,
        usecols=["patientunitstayid", "treatmentoffset", "treatmentstring"],
    )

    mapped_total = 0
    for chunk in reader:
        chunk["patientunitstayid"] = pd.to_numeric(chunk["patientunitstayid"], errors="coerce")
        chunk["treatmentoffset"] = pd.to_numeric(chunk["treatmentoffset"], errors="coerce")
        chunk = chunk.dropna(subset=["patientunitstayid", "treatmentoffset"])
        if chunk.empty:
            continue

        pids = chunk["patientunitstayid"].astype(np.int64).to_numpy()
        times = chunk["treatmentoffset"].astype(np.int64).to_numpy()

        map_df = build_event_to_window_mapping(
            pids=pids,
            times=times,
            sample_index_df=sample_index_df,
            history_mins=history_mins,
            stride_mins=stride_mins,
        )
        if map_df.empty:
            continue

        mapped_total += len(map_df)
        event_i = map_df["event_i"].to_numpy(dtype=np.int32)
        row_idx = map_df["row_idx"].to_numpy(dtype=np.int32)

        np.add.at(tr_cnt, row_idx, 1)

        dialysis_mask = chunk["treatmentstring"].astype(str).str.contains("dialysis", case=False, na=False).to_numpy()
        is_dial = dialysis_mask[event_i]
        if np.any(is_dial):
            dial_any[row_idx[is_dial]] = 1

    print(f"treatment mapped pairs {mapped_total}")
    return pd.DataFrame({
        "treatment_count_in_window": tr_cnt.astype(np.float64),
        "treatment_dialysis_any": dial_any.astype(np.float64),
    })


def process_infusion_vasopressor_any(
    table_path: Path,
    sample_index_df: pd.DataFrame,
    n_samples: int,
    history_mins: int,
    stride_mins: int,
    chunksize: int,
) -> pd.DataFrame:
    if not safe_exists(table_path):
        print(f"WARNING: missing file, skipping: {table_path}")
        return pd.DataFrame({"drug_vasopressor_any": np.zeros(n_samples, dtype=np.float64)})

    vaso_any = np.zeros(n_samples, dtype=np.int32)
    vaso_pattern = (
        "norepi|norad|dopamine|epinephrine|adrenaline|phenylephrine|vasopressin|levophed"
    )

    reader = pd.read_csv(
        table_path,
        compression="infer",
        low_memory=False,
        chunksize=chunksize,
        usecols=["patientunitstayid", "infusionoffset", "drugname"],
    )

    mapped_total = 0
    for chunk in reader:
        chunk["patientunitstayid"] = pd.to_numeric(chunk["patientunitstayid"], errors="coerce")
        chunk["infusionoffset"] = pd.to_numeric(chunk["infusionoffset"], errors="coerce")
        chunk = chunk.dropna(subset=["patientunitstayid", "infusionoffset"])
        if chunk.empty:
            continue

        is_vaso = chunk["drugname"].astype(str).str.contains(vaso_pattern, case=False, na=False).to_numpy()
        if not np.any(is_vaso):
            continue

        chunk = chunk.loc[is_vaso].reset_index(drop=True)

        pids = chunk["patientunitstayid"].astype(np.int64).to_numpy()
        times = chunk["infusionoffset"].astype(np.int64).to_numpy()

        map_df = build_event_to_window_mapping(
            pids=pids,
            times=times,
            sample_index_df=sample_index_df,
            history_mins=history_mins,
            stride_mins=stride_mins,
        )
        if map_df.empty:
            continue

        mapped_total += len(map_df)
        row_idx = map_df["row_idx"].to_numpy(dtype=np.int32)
        vaso_any[row_idx] = 1

    print(f"vasopressor mapped pairs {mapped_total}")
    return pd.DataFrame({"drug_vasopressor_any": vaso_any.astype(np.float64)})


#############################
# Main
#############################
def main() -> None:
    data_dir = config.EICU_DATA_DIR
    samples_csv = config.samples_path(config.DISEASE)
    out_features = config.features_path(config.DISEASE)

    print("#############################")
    print("Aggregating rolling-window features (optimized + chart pivots)")
    print("#############################")
    print(f"Data dir: {data_dir}")
    print(f"Samples: {samples_csv}")
    print(f"Output features: {out_features}")

    samples_df = load_samples(samples_csv)
    sample_index_df = make_sample_index_df(samples_df)

    n_samples = len(samples_df)
    history_mins = config.HISTORY_MINS
    stride_mins = config.STRIDE_MINS

    feats = pd.DataFrame({
        "patientunitstayid": samples_df["patientunitstayid"].values.astype(int),
        "t_end": samples_df["t_end"].values.astype(int),
    })

    feats = pd.concat([feats, build_static_features(data_dir, samples_df)], axis=1)

    # IMPORTANT: remove indirect leakage from Apache-derived features
    drop_prefixes = ("aps_", "apred_", "ares_")
    apache_cols = [c for c in feats.columns if c.startswith(drop_prefixes)]
    if apache_cols:
        print(f"Dropping leak-prone Apache features: {len(apache_cols)} columns")
        feats = feats.drop(columns=apache_cols)

    feats = pd.concat([feats, process_numeric_timeseries_table(
        table_path=data_dir / "vitalPeriodic.csv.gz",
        pid_col="patientunitstayid",
        offset_col="observationoffset",
        usecols=[
            "temperature", "sao2", "heartrate", "respiration",
            "cvp", "etco2",
            "systemicsystolic", "systemicdiastolic", "systemicmean",
            "pasystolic", "padiastolic", "pamean",
            "st1", "st2", "st3", "icp",
        ],
        drop_cols=["vitalperiodicid"],
        prefix="vp_",
        sample_index_df=sample_index_df,
        n_samples=n_samples,
        history_mins=history_mins,
        stride_mins=stride_mins,
        chunksize=1_000_000,
    )], axis=1)

    feats = pd.concat([feats, process_numeric_timeseries_table(
        table_path=data_dir / "vitalAperiodic.csv.gz",
        pid_col="patientunitstayid",
        offset_col="observationoffset",
        usecols=[
            "noninvasivesystolic", "noninvasivediastolic", "noninvasivemean",
            "paop", "cardiacoutput", "cardiacinput", "svr", "svri", "pvr", "pvri",
        ],
        drop_cols=["vitalaperiodicid"],
        prefix="va_",
        sample_index_df=sample_index_df,
        n_samples=n_samples,
        history_mins=history_mins,
        stride_mins=stride_mins,
        chunksize=1_000_000,
    )], axis=1)

    feats = pd.concat([feats, process_numeric_timeseries_table(
        table_path=data_dir / "lab.csv.gz",
        pid_col="patientunitstayid",
        offset_col="labresultoffset",
        usecols=["labresult"],
        drop_cols=["labid"],
        prefix="lab_",
        sample_index_df=sample_index_df,
        n_samples=n_samples,
        history_mins=history_mins,
        stride_mins=stride_mins,
        chunksize=800_000,
    )], axis=1)

    feats = pd.concat([feats, process_numeric_timeseries_table(
        table_path=data_dir / "intakeOutput.csv.gz",
        pid_col="patientunitstayid",
        offset_col="intakeoutputoffset",
        usecols=["intaketotal", "outputtotal", "dialysistotal", "nettotal", "cellvaluenumeric"],
        drop_cols=["intakeoutputid", "cellpath", "celllabel", "cellvaluetext", "intakeoutputentryoffset"],
        prefix="io_",
        sample_index_df=sample_index_df,
        n_samples=n_samples,
        history_mins=history_mins,
        stride_mins=stride_mins,
        chunksize=800_000,
    )], axis=1)

    feats = pd.concat([feats, process_numeric_timeseries_table(
        table_path=data_dir / "respiratoryCare.csv.gz",
        pid_col="patientunitstayid",
        offset_col="respcarestatusoffset",
        usecols=[
            "airwaytype", "airwaysize", "airwayposition", "cuffpressure",
            "apneaparms", "lowexhmvlimit", "hiexhmvlimit", "lowexhtvlimit",
            "hipeakpreslimit", "lowpeakpreslimit", "hirespratelimit", "lowrespratelimit",
            "sighpreslimit", "lowironoxlimit", "highironoxlimit", "meanairwaypreslimit",
            "peeplimit", "cpaplimit", "setapneainterval", "setapneatv",
            "setapneaippeephigh", "setapnearr", "setapneapeakflow",
            "setapneainsptime", "setapneaie", "setapneafio2",
        ],
        drop_cols=["respcareid", "currenthistoryseqnum"],
        prefix="rc_",
        sample_index_df=sample_index_df,
        n_samples=n_samples,
        history_mins=history_mins,
        stride_mins=stride_mins,
        chunksize=800_000,
    )], axis=1)

    feats = pd.concat([feats, process_numeric_timeseries_table(
        table_path=data_dir / "infusionDrug.csv.gz",
        pid_col="patientunitstayid",
        offset_col="infusionoffset",
        usecols=["drugrate", "infusionrate", "drugamount", "volumeoffluid", "patientweight"],
        drop_cols=["infusiondrugid"],
        prefix="inf_",
        sample_index_df=sample_index_df,
        n_samples=n_samples,
        history_mins=history_mins,
        stride_mins=stride_mins,
        chunksize=800_000,
    )], axis=1)

    rch_path = data_dir / "respiratoryCharting.csv.gz"
    rch_labels = find_top_numeric_labels(
        table_path=rch_path,
        label_col="respchartvaluelabel",
        value_col="respchartvalue",
        top_n=int(config.RESPCHART_TOP_LABELS),
        chunksize=800_000,
    )
    print("#############################")
    print(f"RespChart top numeric labels (first 20): {rch_labels[:20]}")
    print("#############################")
    feats = pd.concat([feats, process_charting_pivot(
        table_path=rch_path,
        pid_col="patientunitstayid",
        offset_col="respchartoffset",
        label_col="respchartvaluelabel",
        value_col="respchartvalue",
        prefix="rch_",
        sample_index_df=sample_index_df,
        n_samples=n_samples,
        history_mins=history_mins,
        stride_mins=stride_mins,
        top_labels=rch_labels,
        chunksize=800_000,
    )], axis=1)

    nch_path = data_dir / "nurseCharting.csv.gz"
    nch_labels = find_top_numeric_labels(
        table_path=nch_path,
        label_col="nursingchartcelltypevallabel",
        value_col="nursingchartvalue",
        top_n=int(config.NURSECHART_TOP_LABELS),
        chunksize=800_000,
    )
    print("#############################")
    print(f"NurseChart top numeric labels (first 20): {nch_labels[:20]}")
    print("#############################")
    feats = pd.concat([feats, process_charting_pivot(
        table_path=nch_path,
        pid_col="patientunitstayid",
        offset_col="nursingchartoffset",
        label_col="nursingchartcelltypevallabel",
        value_col="nursingchartvalue",
        prefix="nch_",
        sample_index_df=sample_index_df,
        n_samples=n_samples,
        history_mins=history_mins,
        stride_mins=stride_mins,
        top_labels=nch_labels,
        chunksize=800_000,
    )], axis=1)

    feats = pd.concat([feats, process_count_table(
        table_path=data_dir / "medication.csv.gz",
        pid_col="patientunitstayid",
        offset_col="drugstartoffset",
        sample_index_df=sample_index_df,
        n_samples=n_samples,
        history_mins=history_mins,
        stride_mins=stride_mins,
        chunksize=800_000,
        out_col="med_count_in_window",
    )], axis=1)

    feats = pd.concat([feats, process_treatment_dialysis(
        table_path=data_dir / "treatment.csv.gz",
        sample_index_df=sample_index_df,
        n_samples=n_samples,
        history_mins=history_mins,
        stride_mins=stride_mins,
        chunksize=800_000,
    )], axis=1)

    feats = pd.concat([feats, process_count_table(
        table_path=data_dir / "nurseAssessment.csv.gz",
        pid_col="patientunitstayid",
        offset_col="nurseassessoffset",
        sample_index_df=sample_index_df,
        n_samples=n_samples,
        history_mins=history_mins,
        stride_mins=stride_mins,
        chunksize=800_000,
        out_col="nurseassess_count_in_window",
    )], axis=1)

    feats = pd.concat([feats, process_count_table(
        table_path=data_dir / "nurseCare.csv.gz",
        pid_col="patientunitstayid",
        offset_col="nursecareoffset",
        sample_index_df=sample_index_df,
        n_samples=n_samples,
        history_mins=history_mins,
        stride_mins=stride_mins,
        chunksize=800_000,
        out_col="nursecare_count_in_window",
    )], axis=1)

    feats = pd.concat([feats, process_infusion_vasopressor_any(
        table_path=data_dir / "infusionDrug.csv.gz",
        sample_index_df=sample_index_df,
        n_samples=n_samples,
        history_mins=history_mins,
        stride_mins=stride_mins,
        chunksize=800_000,
    )], axis=1)

    print("#############################")
    print("Saving features parquet")
    print("#############################")
    out_features.parent.mkdir(parents=True, exist_ok=True)
    feats.to_parquet(out_features, index=False)
    print(f"Saved: {out_features}")
    print(f"Final shape: {feats.shape}")


if __name__ == "__main__":
    main()
