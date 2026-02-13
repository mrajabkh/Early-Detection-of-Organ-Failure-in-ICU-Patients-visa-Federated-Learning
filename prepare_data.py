# prepare_data.py
# Build rolling-window samples.csv for a chosen disease definition.
# Adds patient-level train/val/test split to avoid patient overlap across sets.
# Stores t_event (onset time) and lead_time_mins.
#
# Includes:
# - window cap (config-controlled)
# - optional balancing (config-controlled)
# - patient-level split
# - TEST-only monitoring density filter (vp + va presence bins) [MEMORY SAFE]
# - optional negative limiter (NOW applies to train/val/test, per split)
#
# Location: Project/Code/prepare_data.py

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Set

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import config


#############################
# Helpers
#############################
def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def print_counts(df: pd.DataFrame, title: str) -> None:
    n_total = int(len(df))
    n_pos = int(df["label"].sum()) if "label" in df.columns else 0
    n_neg = int((df["label"] == 0).sum()) if "label" in df.columns else 0
    ratio = (n_neg / n_pos) if n_pos > 0 else float("inf")
    prev = (n_pos / n_total) if n_total > 0 else 0.0

    print("#############################")
    print(title)
    print(f"Total windows: {n_total}")
    print(f"Pos windows:   {n_pos}")
    print(f"Neg windows:   {n_neg}")
    print(f"Neg:Pos ratio: {ratio:.2f}:1" if np.isfinite(ratio) else "Neg:Pos ratio: inf")
    print(f"Prevalence:    {prev:.4f}")
    print("#############################")


def read_diagnoses(data_dir: Path) -> pd.DataFrame:
    path = data_dir / "diagnosis.csv.gz"
    df = pd.read_csv(path, compression="infer", low_memory=False)
    required = {"diagnosisstring", "patientunitstayid", "diagnosisoffset"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"diagnosis.csv.gz missing columns: {sorted(missing)}")
    return df


def read_patients(data_dir: Path) -> pd.DataFrame:
    path = data_dir / "patient.csv.gz"
    df = pd.read_csv(path, compression="infer", low_memory=False, usecols=["patientunitstayid"])
    return df


def split_diagnosisstring(diagnoses: pd.DataFrame) -> pd.DataFrame:
    parts = diagnoses["diagnosisstring"].astype(str).str.split("|", n=2, expand=True)
    diagnoses = diagnoses.copy()
    diagnoses["majorcategory"] = parts[0].str.strip()
    diagnoses["subcategory"] = parts[1].str.strip() if parts.shape[1] > 1 else ""
    diagnoses["diagnosisname"] = parts[2].str.strip() if parts.shape[1] > 2 else ""
    return diagnoses


def filter_disease_rows(diagnoses: pd.DataFrame, disease: config.DiseaseSpec) -> pd.DataFrame:
    df = split_diagnosisstring(diagnoses)
    major = disease.major.strip().lower()
    subcat = disease.subcategory.strip().lower() if disease.subcategory else None

    if subcat is None:
        out = df[df["majorcategory"].str.lower() == major]
    else:
        out = df[(df["majorcategory"].str.lower() == major) & (df["subcategory"].str.lower() == subcat)]
    return out


def compute_onset_times(disease_rows: pd.DataFrame) -> pd.Series:
    onset = disease_rows.groupby("patientunitstayid")["diagnosisoffset"].min()
    onset = pd.to_numeric(onset, errors="coerce").dropna()
    onset.index = onset.index.astype(int)
    onset = onset.astype(int)
    return onset


def compute_max_offsets_across_tables(data_dir: Path, chunksize: int = 1_000_000) -> pd.Series:
    sources = [
        ("vitalPeriodic.csv.gz", "observationoffset", ["patientunitstayid", "observationoffset"]),
        ("vitalAperiodic.csv.gz", "observationoffset", ["patientunitstayid", "observationoffset"]),
        ("lab.csv.gz", "labresultoffset", ["patientunitstayid", "labresultoffset"]),
        ("intakeOutput.csv.gz", "intakeoutputoffset", ["patientunitstayid", "intakeoutputoffset"]),
        ("respiratoryCare.csv.gz", "respcarestatusoffset", ["patientunitstayid", "respcarestatusoffset"]),
        ("respiratoryCharting.csv.gz", "respchartoffset", ["patientunitstayid", "respchartoffset"]),
        ("infusionDrug.csv.gz", "infusionoffset", ["patientunitstayid", "infusionoffset"]),
        ("medication.csv.gz", "drugstartoffset", ["patientunitstayid", "drugstartoffset"]),
        ("treatment.csv.gz", "treatmentoffset", ["patientunitstayid", "treatmentoffset"]),
        ("nurseAssessment.csv.gz", "nurseassessoffset", ["patientunitstayid", "nurseassessoffset"]),
        ("nurseCare.csv.gz", "nursecareoffset", ["patientunitstayid", "nursecareoffset"]),
        ("nurseCharting.csv.gz", "nursingchartoffset", ["patientunitstayid", "nursingchartoffset"]),
    ]

    max_dict: Dict[int, int] = defaultdict(int)

    for fname, offset_col, usecols in sources:
        path = data_dir / fname
        if not path.exists():
            print(f"WARNING: missing file for coverage, skipping: {fname}")
            continue

        print("#############################")
        print(f"Coverage scan: {fname} ({offset_col})")
        print("#############################")

        reader = pd.read_csv(
            path,
            compression="infer",
            usecols=usecols,
            chunksize=chunksize,
            low_memory=False,
        )

        for chunk in reader:
            if "patientunitstayid" not in chunk.columns or offset_col not in chunk.columns:
                continue

            chunk["patientunitstayid"] = pd.to_numeric(chunk["patientunitstayid"], errors="coerce")
            chunk[offset_col] = pd.to_numeric(chunk[offset_col], errors="coerce")
            chunk = chunk.dropna(subset=["patientunitstayid", offset_col])
            if chunk.empty:
                continue

            chunk["patientunitstayid"] = chunk["patientunitstayid"].astype(int)
            grp = chunk.groupby("patientunitstayid")[offset_col].max()

            for pid, max_off in grp.items():
                if pd.isna(max_off):
                    continue
                max_off_int = int(max_off)
                if max_off_int > max_dict[pid]:
                    max_dict[pid] = max_off_int

    return pd.Series(max_dict, name="max_offset_any_source")


def generate_patient_windows(
    patient_id: int,
    max_offset: int,
    onset_time: Optional[int],
    history_mins: int,
    horizon_mins: int,
    stride_mins: int,
    require_full_horizon: bool,
    stop_after_event: bool,
) -> List[Tuple[int, int, int, float, float]]:
    last_t_end = max_offset
    if require_full_horizon:
        last_t_end = max_offset - horizon_mins

    if last_t_end < history_mins:
        return []

    t_ends = list(range(history_mins, last_t_end + 1, stride_mins))

    t_event = float(onset_time) if onset_time is not None else float("nan")

    rows: List[Tuple[int, int, int, float, float]] = []
    for t_end in t_ends:
        if onset_time is not None and stop_after_event and t_end >= onset_time:
            break

        label = 0
        if onset_time is not None:
            if (t_end < onset_time) and (onset_time <= t_end + horizon_mins):
                label = 1

        lead_time = float(onset_time - t_end) if onset_time is not None else float("nan")
        rows.append((patient_id, t_end, label, t_event, lead_time))

    return rows


def add_patient_level_splits(samples_df: pd.DataFrame) -> pd.DataFrame:
    per_patient = (
        samples_df.groupby("patientunitstayid")["label"]
        .max()
        .astype(int)
        .rename("patient_has_positive")
        .reset_index()
    )

    pids = per_patient["patientunitstayid"].to_numpy()
    strat = per_patient["patient_has_positive"].to_numpy()

    test_size = float(getattr(config, "TEST_SIZE", 0.2))
    val_size = float(getattr(config, "VAL_SIZE", 0.15))
    rs = int(getattr(config, "SPLIT_RANDOM_STATE", 42))

    trainval_pids, test_pids = train_test_split(
        pids,
        test_size=test_size,
        random_state=rs,
        stratify=strat,
    )

    trainval_strat = per_patient.set_index("patientunitstayid").loc[trainval_pids, "patient_has_positive"].to_numpy()

    trainval_frac = 1.0 - test_size
    val_frac_of_trainval = val_size / trainval_frac
    val_frac_of_trainval = float(np.clip(val_frac_of_trainval, 0.01, 0.8))

    train_pids, val_pids = train_test_split(
        trainval_pids,
        test_size=val_frac_of_trainval,
        random_state=rs,
        stratify=trainval_strat,
    )

    train_set = set(int(x) for x in train_pids.tolist())
    val_set = set(int(x) for x in val_pids.tolist())

    out = samples_df.copy()
    out["split"] = "test"
    out.loc[out["patientunitstayid"].isin(train_set), "split"] = "train"
    out.loc[out["patientunitstayid"].isin(val_set), "split"] = "val"
    return out


#############################
# Window cap (config-controlled)
#############################
def _cap_both_patient_groups(samples_df: pd.DataFrame, max_windows: int, seed: int) -> pd.DataFrame:
    total_windows = int(len(samples_df))
    if total_windows <= max_windows:
        return samples_df

    per_patient = samples_df.groupby("patientunitstayid").agg(
        n_windows=("label", "size"),
        has_pos=("label", "max"),
    ).reset_index()
    per_patient["has_pos"] = per_patient["has_pos"].astype(int)

    total_windows_now = int(per_patient["n_windows"].sum())
    if total_windows_now <= max_windows:
        return samples_df

    p = float(max_windows) / float(total_windows_now)
    p = float(np.clip(p, 0.0, 1.0))

    rng = np.random.RandomState(seed)

    event = per_patient[per_patient["has_pos"] == 1]
    nonev = per_patient[per_patient["has_pos"] == 0]

    event_keep = rng.rand(len(event)) < p
    nonev_keep = rng.rand(len(nonev)) < p

    keep_pids = set(event.loc[event_keep, "patientunitstayid"].astype(int).tolist())
    keep_pids.update(nonev.loc[nonev_keep, "patientunitstayid"].astype(int).tolist())

    out = samples_df[samples_df["patientunitstayid"].isin(keep_pids)].copy()

    out_windows = int(len(out))
    if out_windows > max_windows:
        kept = per_patient[per_patient["patientunitstayid"].isin(keep_pids)].copy()
        drop_pool = kept[kept["has_pos"] == 0].sample(frac=1.0, random_state=seed)

        cur_keep = set(keep_pids)
        cur_windows = out_windows
        for _, row in drop_pool.iterrows():
            pid = int(row["patientunitstayid"])
            w = int(row["n_windows"])
            if cur_windows - w >= max_windows:
                cur_keep.remove(pid)
                cur_windows -= w
            if cur_windows <= max_windows:
                break

        out = samples_df[samples_df["patientunitstayid"].isin(cur_keep)].copy()

    return out


def _cap_neg_only_patients(samples_df: pd.DataFrame, max_windows: int, seed: int) -> pd.DataFrame:
    total_windows = int(len(samples_df))
    if total_windows <= max_windows:
        return samples_df

    per_patient = samples_df.groupby("patientunitstayid").agg(
        n_windows=("label", "size"),
        has_pos=("label", "max"),
    ).reset_index()
    per_patient["has_pos"] = per_patient["has_pos"].astype(int)

    event = per_patient[per_patient["has_pos"] == 1].copy()
    nonev = per_patient[per_patient["has_pos"] == 0].copy()

    event_pids = event["patientunitstayid"].astype(int).tolist()
    event_windows = int(event["n_windows"].sum())

    if event_windows >= max_windows:
        keep_pids = set(event_pids)
        return samples_df[samples_df["patientunitstayid"].isin(keep_pids)].copy()

    budget_for_nonev = int(max_windows - event_windows)

    nonev_shuf = nonev.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    keep_nonev: List[int] = []
    used = 0
    for _, row in nonev_shuf.iterrows():
        pid = int(row["patientunitstayid"])
        w = int(row["n_windows"])
        if used + w > budget_for_nonev:
            continue
        keep_nonev.append(pid)
        used += w
        if used >= budget_for_nonev:
            break

    keep_pids = set(event_pids) | set(keep_nonev)
    return samples_df[samples_df["patientunitstayid"].isin(keep_pids)].copy()


def apply_window_cap_from_config(samples_df: pd.DataFrame) -> pd.DataFrame:
    enabled = bool(getattr(config, "WINDOW_CAP_ENABLED", False))
    max_windows = int(getattr(config, "MAX_WINDOWS", 0))
    mode = str(getattr(config, "WINDOW_CAP_MODE", "both")).strip().lower()
    rs = int(getattr(config, "WINDOW_CAP_RANDOM_STATE", getattr(config, "SEED", 42)))

    if (not enabled) or max_windows <= 0:
        return samples_df

    print("#############################")
    print("Applying patient-level window cap (config-controlled)")
    print(f"WINDOW_CAP_ENABLED: {enabled}")
    print(f"MAX_WINDOWS: {max_windows}")
    print(f"WINDOW_CAP_MODE: {mode}")
    print(f"WINDOW_CAP_RANDOM_STATE: {rs}")
    print("#############################")

    if mode == "neg_only":
        return _cap_neg_only_patients(samples_df, max_windows=max_windows, seed=rs)
    if mode == "both":
        return _cap_both_patient_groups(samples_df, max_windows=max_windows, seed=rs)

    print("#############################")
    print(f"WARNING: Unknown WINDOW_CAP_MODE: {mode}. No cap applied.")
    print("#############################")
    return samples_df


#############################
# Optional balancing (config-controlled)
#############################
def apply_balancing_from_config(samples_df: pd.DataFrame) -> pd.DataFrame:
    enabled = bool(getattr(config, "BALANCE_ENABLED", False))
    if not enabled:
        return samples_df

    ratio = float(getattr(config, "NEGATIVE_POSITIVE_RATIO", 1.0))
    cap = getattr(config, "NEGATIVE_WINDOWS_PER_PATIENT_CAP", None)

    print("#############################")
    print("Balancing negatives using NEGATIVE_POSITIVE_RATIO")
    print(f"BALANCE_ENABLED: {enabled}")
    print(f"NEGATIVE_POSITIVE_RATIO: {ratio}")
    print(f"NEGATIVE_WINDOWS_PER_PATIENT_CAP: {cap}")
    print("#############################")

    n_pos = int(samples_df["label"].sum())
    if n_pos == 0:
        raise RuntimeError("No positive samples; cannot balance.")

    pos_df = samples_df[samples_df["label"] == 1].copy()
    neg_df = samples_df[samples_df["label"] == 0].copy()

    if cap is not None:
        cap = int(cap)
        neg_df = (
            neg_df.groupby("patientunitstayid", group_keys=False)
            .apply(lambda g: g.sample(n=min(len(g), cap), random_state=config.SEED))
            .reset_index(drop=True)
        )

    target_neg = int(round(ratio * len(pos_df)))
    target_neg = min(target_neg, len(neg_df))

    neg_df = neg_df.sample(n=target_neg, random_state=config.SEED).reset_index(drop=True)

    out = (
        pd.concat([pos_df, neg_df], axis=0)
        .sample(frac=1.0, random_state=config.SEED)
        .reset_index(drop=True)
    )
    return out


#############################
# TEST-only monitoring density filter (vp + va) [MEMORY SAFE]
#############################
def _ceil_to_stride(arr: np.ndarray, stride_mins: int) -> np.ndarray:
    return ((arr + stride_mins - 1) // stride_mins) * stride_mins


def _encode_pid_bin(pid: np.ndarray, bin_end: np.ndarray) -> np.ndarray:
    """
    Encode (pid, bin_end) to uint64 key:
      key = (pid << 32) | (bin_end & 0xFFFFFFFF)
    Assumes pid and bin_end fit in 32 bits (true for eICU ids + minute offsets).
    """
    pid_u = pid.astype(np.uint64)
    be_u = (bin_end.astype(np.uint64) & np.uint64(0xFFFFFFFF))
    return (pid_u << np.uint64(32)) | be_u


def _build_presence_keyset_for_pids(
    data_dir: Path,
    pids_keep: Set[int],
    stride_mins: int,
    chunksize: int = 800_000,
) -> Set[int]:
    """
    Build a set of uint64-encoded keys (pid, bin_end) where ANY vitals is present in that bin.
    Only for patients in pids_keep. Does NOT build a giant dataframe and does NOT drop_duplicates.
    """
    keyset: Set[int] = set()

    vp_cols = list(getattr(config, "VP_VITAL_COLS", []))
    va_cols = list(getattr(config, "VA_VITAL_COLS", []))

    def _process_file(path: Path, offset_col: str, cols: List[str]) -> None:
        if not path.exists():
            return

        usecols = ["patientunitstayid", offset_col] + cols
        usecols = list(dict.fromkeys(usecols))

        reader = pd.read_csv(path, compression="infer", low_memory=False, usecols=usecols, chunksize=chunksize)

        for chunk in reader:
            chunk["patientunitstayid"] = pd.to_numeric(chunk["patientunitstayid"], errors="coerce")
            chunk[offset_col] = pd.to_numeric(chunk[offset_col], errors="coerce")
            chunk = chunk.dropna(subset=["patientunitstayid", offset_col])
            if chunk.empty:
                continue

            pid = chunk["patientunitstayid"].astype(np.int64)
            mask_pid = pid.isin(pids_keep)
            if not mask_pid.any():
                continue
            chunk = chunk.loc[mask_pid].copy()
            if chunk.empty:
                continue

            pid_arr = chunk["patientunitstayid"].astype(np.int64).to_numpy()
            t_arr = chunk[offset_col].astype(np.int64).to_numpy()

            cols_present = [c for c in cols if c in chunk.columns]
            if cols_present:
                any_pres = chunk[cols_present].notna().any(axis=1).to_numpy()
            else:
                any_pres = np.ones(len(chunk), dtype=bool)

            if not np.any(any_pres):
                continue

            pid_arr = pid_arr[any_pres]
            t_arr = t_arr[any_pres]

            bin_end = _ceil_to_stride(t_arr, stride_mins)
            keys = _encode_pid_bin(pid_arr, bin_end)

            keyset.update(int(k) for k in keys.tolist())

    _process_file(data_dir / "vitalPeriodic.csv.gz", "observationoffset", vp_cols)
    _process_file(data_dir / "vitalAperiodic.csv.gz", "observationoffset", va_cols)

    return keyset


def _compute_density_for_rows(
    df: pd.DataFrame,
    keyset: Set[int],
    stride_mins: int,
    history_mins: int,
) -> np.ndarray:
    """
    Density = number of history bins with ANY vitals present.
    Uses set membership lookups: O(n_rows * (history_mins/stride_mins)).
    """
    offsets = np.arange(0, history_mins, stride_mins, dtype=np.int64)
    pid = df["patientunitstayid"].astype(np.int64).to_numpy()
    t_end = df["t_end"].astype(np.int64).to_numpy()

    dens = np.zeros(len(df), dtype=np.int32)

    for off in offsets:
        be = t_end - off
        keys = _encode_pid_bin(pid, be)
        dens += np.fromiter((1 if int(k) in keyset else 0 for k in keys), count=len(keys), dtype=np.int32)

    return dens


def apply_test_density_filter(samples_df: pd.DataFrame) -> pd.DataFrame:
    enabled = bool(getattr(config, "APPLY_TEST_DENSITY_FILTER", False))
    if not enabled:
        return samples_df

    if "split" not in samples_df.columns:
        print("#############################")
        print("WARNING: No split column found; cannot apply test density filter.")
        print("#############################")
        return samples_df

    stride_mins = int(getattr(config, "STRIDE_MINS", 60))
    history_mins = int(getattr(config, "HISTORY_MINS", 720))
    q = float(getattr(config, "TEST_DENSITY_POS_QUANTILE", 0.25))

    print("#############################")
    print("Applying TEST-only monitoring density filter")
    print("Definition: count of hourly bins in history with ANY vitals present (vp or va)")
    print(f"TEST_DENSITY_POS_QUANTILE: {q}")
    print("#############################")

    train_pos = samples_df[(samples_df["split"] == "train") & (samples_df["label"] == 1)].copy()
    test_df = samples_df[samples_df["split"] == "test"].copy()

    if train_pos.empty or test_df.empty:
        print("#############################")
        print("WARNING: train_pos or test_df empty; skipping density filter.")
        print("#############################")
        return samples_df

    pids_needed = set(train_pos["patientunitstayid"].astype(int).unique().tolist())
    pids_needed.update(test_df["patientunitstayid"].astype(int).unique().tolist())

    print("#############################")
    print(f"Building vitals presence keyset for patients: {len(pids_needed)}")
    print("#############################")

    keyset = _build_presence_keyset_for_pids(
        data_dir=config.EICU_DATA_DIR,
        pids_keep=pids_needed,
        stride_mins=stride_mins,
        chunksize=800_000,
    )

    print("#############################")
    print(f"Presence keyset size: {len(keyset)}")
    print("#############################")

    train_pos_density = _compute_density_for_rows(train_pos, keyset, stride_mins=stride_mins, history_mins=history_mins)
    thr = float(np.quantile(train_pos_density.astype(np.float64), q))

    print("#############################")
    print(f"Density threshold (train pos quantile): {thr:.3f}")
    print("#############################")

    test_density = _compute_density_for_rows(test_df, keyset, stride_mins=stride_mins, history_mins=history_mins)
    keep_test = test_density >= thr

    before_test = int(len(test_df))
    after_test = int(keep_test.sum())

    print("#############################")
    print(f"TEST rows before density filter: {before_test}")
    print(f"TEST rows after density filter:  {after_test}")
    print("#############################")

    kept_test_ids = test_df.index.to_numpy()[keep_test]
    df2 = pd.concat(
        [
            samples_df[samples_df["split"] != "test"],
            samples_df.loc[kept_test_ids],
        ],
        axis=0,
    ).reset_index(drop=True)

    return df2


#############################
# Optional negative limiter applied to ALL splits (per split)
#############################
def apply_neg_limiter_all_splits(samples_df: pd.DataFrame) -> pd.DataFrame:
    enabled = bool(getattr(config, "TEST_NEG_LIMITER_ENABLED", False))
    if not enabled:
        return samples_df

    if "split" not in samples_df.columns:
        return samples_df

    max_ratio = float(getattr(config, "TEST_NEG_POS_MAX_RATIO", 97.0))
    rs = int(getattr(config, "TEST_NEG_LIMITER_RANDOM_STATE", 42))

    print("#############################")
    print("Applying negative limiter to ALL splits (per split)")
    print("Rule: keep all positives, downsample negatives to Neg <= max_ratio * Pos")
    print(f"MAX_RATIO (Neg:Pos): {max_ratio}")
    print(f"RANDOM_STATE: {rs}")
    print("#############################")

    out_blocks: List[pd.DataFrame] = []
    df = samples_df.copy()

    for sp in ["train", "val", "test"]:
        part = df[df["split"] == sp].copy()
        if part.empty:
            continue

        n_pos = int(part["label"].sum())
        if n_pos == 0:
            # Nothing to ratio-limit against; keep as-is
            out_blocks.append(part)
            continue

        pos_df = part[part["label"] == 1]
        neg_df = part[part["label"] == 0]

        max_neg = int(round(max_ratio * len(pos_df)))
        if len(neg_df) <= max_neg:
            out_blocks.append(part)
            continue

        neg_keep = neg_df.sample(n=max_neg, random_state=rs)
        new_part = (
            pd.concat([pos_df, neg_keep], axis=0)
            .sample(frac=1.0, random_state=rs)
            .reset_index(drop=True)
        )
        out_blocks.append(new_part)

        print("#############################")
        print(f"Limiter applied to split={sp}")
        print(f"Pos: {len(pos_df)} | Neg before: {len(neg_df)} | Neg after: {len(neg_keep)}")
        print("#############################")

    # Keep any unexpected split labels as-is
    other = df[~df["split"].isin(["train", "val", "test"])].copy()
    if not other.empty:
        out_blocks.append(other)

    return pd.concat(out_blocks, axis=0).reset_index(drop=True)


#############################
# Main
#############################
def main() -> None:
    set_seeds(int(getattr(config, "SEED", 42)))

    data_dir = config.EICU_DATA_DIR
    out_samples = config.samples_path(config.DISEASE)
    out_meta = config.run_dir(config.DISEASE) / f"meta__{config.disease_tag(config.DISEASE)}.json"

    print("#############################")
    print("Preparing rolling-window samples")
    print("#############################")
    print(f"Data dir: {data_dir}")
    print(f"Output samples: {out_samples}")

    print("#############################")
    print("Loading diagnosis and patient tables")
    print("#############################")
    diagnoses = read_diagnoses(data_dir)
    patients = read_patients(data_dir)
    all_patients = patients["patientunitstayid"].astype(int).unique().tolist()
    print(f"Total ICU stays in patient.csv.gz: {len(all_patients)}")

    print("#############################")
    print("Computing max coverage per patient (across multiple tables)")
    print("#############################")
    max_offsets = compute_max_offsets_across_tables(data_dir)
    print(f"Patients with any coverage in selected tables: {len(max_offsets)}")

    print("#############################")
    print("Finding disease onset times")
    print("#############################")
    disease_rows = filter_disease_rows(diagnoses, config.DISEASE)
    onset_times = compute_onset_times(disease_rows)
    print(f"Patients with at least one matching diagnosis: {len(onset_times)}")

    print("#############################")
    print("Generating rolling windows")
    print("#############################")
    history_mins = int(config.HISTORY_MINS)
    horizon_mins = int(config.HORIZON_MINS)
    stride_mins = int(config.STRIDE_MINS)
    require_full_horizon = bool(config.REQUIRE_FULL_HORIZON)
    stop_after_event = True

    all_rows: List[Tuple[int, int, int, float, float]] = []
    eligible_patients = [pid for pid in all_patients if pid in max_offsets.index]
    print(f"Patients present in coverage index: {len(eligible_patients)}")

    for pid in eligible_patients:
        max_off = int(max_offsets.loc[pid])
        onset = int(onset_times.loc[pid]) if pid in onset_times.index else None
        rows = generate_patient_windows(
            patient_id=pid,
            max_offset=max_off,
            onset_time=onset,
            history_mins=history_mins,
            horizon_mins=horizon_mins,
            stride_mins=stride_mins,
            require_full_horizon=require_full_horizon,
            stop_after_event=stop_after_event,
        )
        if rows:
            all_rows.extend(rows)

    samples_df = pd.DataFrame(all_rows, columns=["patientunitstayid", "t_end", "label", "t_event", "lead_time_mins"])
    if samples_df.empty:
        raise RuntimeError("No samples were generated. Check disease filter and coverage rules.")

    print_counts(samples_df, "Generated samples (pre-cap, pre-balance, pre-split)")

    #############################
    # Cap windows (optional)
    #############################
    samples_df = apply_window_cap_from_config(samples_df)
    print_counts(samples_df, "After window cap (if enabled)")

    #############################
    # Optional balancing (global)
    #############################
    samples_df = apply_balancing_from_config(samples_df)
    print_counts(samples_df, "After balancing (if enabled)")

    #############################
    # Patient-level split
    #############################
    print("#############################")
    print("Assigning patient-level train/val/test split")
    print("#############################")
    samples_df = add_patient_level_splits(samples_df)

    #############################
    # Test-only density filter
    #############################
    samples_df = apply_test_density_filter(samples_df)

    #############################
    # Negative limiter on ALL splits (per split)
    #############################
    samples_df = apply_neg_limiter_all_splits(samples_df)

    #############################
    # Final per-split counts
    #############################
    for sp in ["train", "val", "test"]:
        sub = samples_df[samples_df["split"] == sp]
        print_counts(sub, f"Split={sp} FINAL")

    #############################
    # Save
    #############################
    print("#############################")
    print("Saving samples and meta")
    print("#############################")
    out_samples.parent.mkdir(parents=True, exist_ok=True)
    samples_df.to_csv(out_samples, index=False)

    meta = {
        "disease": {"major": config.DISEASE.major, "subcategory": config.DISEASE.subcategory},
        "rolling_window": {
            "history_hours": config.HISTORY_HRS,
            "horizon_hours": config.HORIZON_HRS,
            "stride_minutes": config.STRIDE_MINS,
            "require_full_horizon": config.REQUIRE_FULL_HORIZON,
            "stop_after_event": stop_after_event,
        },
        "window_cap": {
            "enabled": bool(getattr(config, "WINDOW_CAP_ENABLED", False)),
            "max_windows": int(getattr(config, "MAX_WINDOWS", 0)),
            "mode": str(getattr(config, "WINDOW_CAP_MODE", "both")),
            "random_state": int(getattr(config, "WINDOW_CAP_RANDOM_STATE", getattr(config, "SEED", 42))),
        },
        "balancing": {
            "enabled": bool(getattr(config, "BALANCE_ENABLED", False)),
            "neg_pos_ratio": float(getattr(config, "NEGATIVE_POSITIVE_RATIO", 1.0)),
            "neg_windows_per_patient_cap": getattr(config, "NEGATIVE_WINDOWS_PER_PATIENT_CAP", None),
        },
        "test_density_filter": {
            "enabled": bool(getattr(config, "APPLY_TEST_DENSITY_FILTER", False)),
            "pos_quantile": float(getattr(config, "TEST_DENSITY_POS_QUANTILE", 0.25)),
        },
        "neg_limiter_all_splits": {
            "enabled": bool(getattr(config, "TEST_NEG_LIMITER_ENABLED", False)),
            "max_ratio": float(getattr(config, "TEST_NEG_POS_MAX_RATIO", 97.0)),
            "random_state": int(getattr(config, "TEST_NEG_LIMITER_RANDOM_STATE", 42)),
        },
        "split": {
            "type": "patient_level_train_val_test",
            "test_size": float(getattr(config, "TEST_SIZE", 0.2)),
            "val_size": float(getattr(config, "VAL_SIZE", 0.15)),
            "random_state": int(getattr(config, "SPLIT_RANDOM_STATE", 42)),
        },
        "seed": int(getattr(config, "SEED", 42)),
        "counts": {
            "final_total": int(len(samples_df)),
            "final_pos": int(samples_df["label"].sum()),
            "final_neg": int((samples_df["label"] == 0).sum()),
            "train_patients": int(samples_df.loc[samples_df["split"] == "train", "patientunitstayid"].nunique()),
            "val_patients": int(samples_df.loc[samples_df["split"] == "val", "patientunitstayid"].nunique()),
            "test_patients": int(samples_df.loc[samples_df["split"] == "test", "patientunitstayid"].nunique()),
        },
    }

    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("Done.")
    print(f"Saved samples: {out_samples}")
    print(f"Saved meta: {out_meta}")


if __name__ == "__main__":
    main()
