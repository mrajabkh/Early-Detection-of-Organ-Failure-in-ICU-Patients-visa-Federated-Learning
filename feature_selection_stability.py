# feature_selection_stability.py
# RF-only stability feature selection (memory-safe for 8GB RAM).
# Produces a single CSV at stability_combined_path() (compatible with top-K pipeline).
#
# Key fix: read features.parquet in COLUMN BATCHES using pyarrow, so we never load the full
# wide table into pandas at once (prevents ArrowMemoryError).
#
# IMPORTANT:
# - Missingness mask features (e.g. vp_sao2_max_missing) should be created upstream and saved
#   into features.parquet by aggregate_features.py.
# - Therefore we do NOT create missingness masks in this script anymore. This script ranks
#   exactly what exists in features.parquet.
#
# Location: Project/Code/feature_selection_stability.py

from __future__ import annotations

import time
from typing import List, Tuple

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

import pyarrow.parquet as pq

import config


#############################
# Utilities
#############################
def set_seeds(seed: int) -> None:
    np.random.seed(seed)


def _topk_freq_from_matrix(importance_mat: np.ndarray, feature_names: List[str], topk_ref: int) -> pd.Series:
    n_runs, n_feat = importance_mat.shape
    k = min(int(topk_ref), n_feat)

    counts = np.zeros(n_feat, dtype=np.int32)
    for i in range(n_runs):
        idx = np.argsort(-importance_mat[i])[:k]
        counts[idx] += 1

    return pd.Series(counts / float(n_runs), index=feature_names, name="rf_stability_freq")


def _minmax_norm(s: pd.Series) -> pd.Series:
    mn = float(s.min())
    mx = float(s.max())
    if mx <= mn:
        return pd.Series(np.zeros(len(s), dtype=np.float64), index=s.index)
    return (s - mn) / (mx - mn)


def _replace_inf_with_nan_numeric_inplace(df: pd.DataFrame, chunk_cols: int = 128) -> None:
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not num_cols:
        return
    for i in range(0, len(num_cols), chunk_cols):
        cols = num_cols[i : i + chunk_cols]
        block = df[cols].to_numpy(copy=False)
        if np.issubdtype(block.dtype, np.floating):
            bad = ~np.isfinite(block)
            if bad.any():
                block[bad] = np.nan


#############################
# Load TRAIN + subsample
#############################
def _load_train_samples(samples_path: str) -> pd.DataFrame:
    df = pd.read_csv(samples_path)

    need = {"patientunitstayid", "t_end", "label", "split"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"samples.csv missing columns: {sorted(missing)}")

    df["split"] = df["split"].astype(str).str.lower()
    df = df[df["split"] == "train"].copy()

    df["patientunitstayid"] = pd.to_numeric(df["patientunitstayid"], errors="coerce")
    df["t_end"] = pd.to_numeric(df["t_end"], errors="coerce")
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df.dropna(subset=["patientunitstayid", "t_end", "label"]).copy()

    df["patientunitstayid"] = df["patientunitstayid"].astype(np.int64)
    df["t_end"] = df["t_end"].astype(np.int64)
    df["label"] = df["label"].astype(np.int64)

    return df[["patientunitstayid", "t_end", "label"]]


def _subsample_train_windows(train_df: pd.DataFrame) -> pd.DataFrame:
    max_rows = int(getattr(config, "FS_MAX_TRAIN_ROWS", 70_000))
    rs = int(getattr(config, "FS_RANDOM_STATE", getattr(config, "SEED", 42)))
    strat = bool(getattr(config, "FS_STRATIFIED_SUBSAMPLE", True))

    if max_rows <= 0 or len(train_df) <= max_rows:
        return train_df

    if strat:
        y = train_df["label"].to_numpy()
        idx_all = np.arange(len(train_df))
        idx_sub, _ = train_test_split(
            idx_all,
            train_size=max_rows,
            random_state=rs,
            stratify=y,
        )
        return train_df.iloc[idx_sub].copy()

    return train_df.sample(n=max_rows, random_state=rs).copy()


#############################
# Column-batched parquet read + merge
#############################
def _load_features_for_keys_batched(features_path: str, keys_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a merged dataframe for the sampled keys without loading the full parquet into RAM.

    Strategy:
      - Read only key columns + a batch of feature columns
      - Merge onto keys_df
      - Accumulate features aligned to keys_df order
    """
    keys = keys_df[["patientunitstayid", "t_end", "label"]].copy()
    keys["__row__"] = np.arange(len(keys), dtype=np.int64)

    pf = pq.ParquetFile(features_path)
    all_cols = pf.schema.names

    key_cols = ["patientunitstayid", "t_end"]
    for c in key_cols:
        if c not in all_cols:
            raise ValueError(f"features.parquet missing key column: {c}")

    feat_cols = [c for c in all_cols if c not in key_cols]

    batch_size = int(getattr(config, "FS_PARQUET_COL_BATCH", 50))
    batch_size = max(10, min(batch_size, 200))

    print("#############################")
    print("Reading features.parquet in column batches")
    print(f"Total feature cols in parquet: {len(feat_cols)}")
    print(f"Batch size: {batch_size}")
    print("#############################")

    out_cols: List[str] = []
    out_data: List[np.ndarray] = []

    for start in range(0, len(feat_cols), batch_size):
        batch = feat_cols[start : start + batch_size]
        cols_to_read = key_cols + batch

        t0 = time.time()
        table = pf.read(columns=cols_to_read)
        df_chunk = table.to_pandas(self_destruct=True)

        df_chunk["patientunitstayid"] = pd.to_numeric(df_chunk["patientunitstayid"], errors="coerce")
        df_chunk["t_end"] = pd.to_numeric(df_chunk["t_end"], errors="coerce")
        df_chunk = df_chunk.dropna(subset=["patientunitstayid", "t_end"])
        df_chunk["patientunitstayid"] = df_chunk["patientunitstayid"].astype(np.int64)
        df_chunk["t_end"] = df_chunk["t_end"].astype(np.int64)

        _replace_inf_with_nan_numeric_inplace(df_chunk, chunk_cols=64)

        merged = keys.merge(df_chunk, on=["patientunitstayid", "t_end"], how="left", sort=False, copy=False)
        merged = merged.sort_values("__row__")
        merged = merged[batch]

        for c in batch:
            if c in merged.columns:
                if merged[c].dtype == object:
                    merged[c] = pd.to_numeric(merged[c], errors="coerce")

        for c in batch:
            if c in merged.columns and merged[c].dtype == np.float64:
                merged[c] = merged[c].astype(np.float32)

        out_cols.extend(batch)
        out_data.append(merged.to_numpy(copy=False))

        print(f"Batch {start // batch_size + 1}: cols {len(batch)} loaded in {time.time() - t0:.2f}s")

    X_mat = np.concatenate(out_data, axis=1) if out_data else np.zeros((len(keys), 0), dtype=np.float32)
    out_df = pd.DataFrame(X_mat, columns=out_cols)
    out_df.insert(0, "label", keys["label"].to_numpy(dtype=np.int64))

    return out_df


def _build_Xy_from_Xdf(X_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    y = X_df["label"].to_numpy(dtype=np.int64)
    feats = X_df.drop(columns=["label"])

    # Drop all-null columns
    all_null_cols = feats.columns[feats.isna().all(axis=0)].tolist()
    if all_null_cols:
        print("#############################")
        print("Dropping all-null feature columns (no observed values in subsample)")
        print(f"Dropped: {len(all_null_cols)}")
        print(f"Examples: {all_null_cols[:20]}")
        print("#############################")
        feats = feats.drop(columns=all_null_cols)

    feature_names = feats.columns.astype(str).tolist()

    imputer = SimpleImputer(strategy=getattr(config, "IMPUTE_STRATEGY", "median"))
    X_imp = imputer.fit_transform(feats)

    if X_imp.shape[1] != len(feature_names):
        raise RuntimeError(
            f"Feature name mismatch after imputation: X has {X_imp.shape[1]} cols, "
            f"feature_names has {len(feature_names)}"
        )

    return X_imp.astype(np.float32, copy=False), y, feature_names


#############################
# Main
#############################
def main() -> None:
    set_seeds(int(getattr(config, "SEED", 42)))

    features_path = str(config.features_path(config.DISEASE))
    samples_path = str(config.samples_path(config.DISEASE))
    out_combined = config.stability_combined_path(config.DISEASE)

    print("#############################")
    print("RF-only stability feature selection (single output CSV)")
    print("#############################")
    print(f"Features: {features_path}")
    print(f"Samples:  {samples_path}")
    print(f"Out Comb: {out_combined}")

    print("#############################")
    print("Loading TRAIN split keys")
    print("#############################")
    train_samples = _load_train_samples(samples_path)

    print(f"Train windows (raw):   {len(train_samples)}")
    print(f"Train positives (raw): {int(train_samples['label'].sum())}")
    print(f"Train negatives (raw): {int((train_samples['label'] == 0).sum())}")

    print("#############################")
    print("Subsampling TRAIN windows for feature selection")
    print("#############################")
    train_sub = _subsample_train_windows(train_samples)

    print(f"Train windows (sub):   {len(train_sub)}")
    print(f"Train positives (sub): {int(train_sub['label'].sum())}")
    print(f"Train negatives (sub): {int((train_sub['label'] == 0).sum())}")

    print("#############################")
    print("Building X_df via batched parquet reads")
    print("#############################")
    X_df = _load_features_for_keys_batched(features_path, train_sub)

    print("#############################")
    print("Imputing and building X/y")
    print("#############################")
    X, y, feat_names = _build_Xy_from_Xdf(X_df)

    n_feat = len(feat_names)
    print(f"X shape:  {X.shape}")
    print(f"Features: {n_feat}")

    idx_all = np.arange(len(y), dtype=np.int32)
    idx_boot_pool, _ = train_test_split(
        idx_all,
        test_size=0.25,
        random_state=int(getattr(config, "SEED", 42)),
        stratify=y if bool(getattr(config, "STRATIFY_SPLIT", True)) else None,
    )

    print("#############################")
    print(f"Boot pool rows: {len(idx_boot_pool)}")
    print("#############################")

    n_runs = int(getattr(config, "STAB_N_BOOTSTRAPS", 15))
    boot_frac = float(getattr(config, "STAB_BOOTSTRAP_FRAC", 0.7))
    topk_ref = int(getattr(config, "STAB_TOPK_REF", 100))
    freq_thr = float(getattr(config, "STAB_FREQ_THRESHOLD", 0.6))

    rf_params = dict(getattr(config, "RF_TUNED_PARAMS", {}))
    rf_importance_mat = np.zeros((n_runs, n_feat), dtype=np.float64)

    for r in range(n_runs):
        print("#############################")
        print(f"RF Bootstrap {r + 1}/{n_runs}")
        print("#############################")

        rs = np.random.RandomState(int(getattr(config, "SEED", 42)) + 1000 + r)

        boot_n = max(10, int(round(boot_frac * len(idx_boot_pool))))
        boot_idx = rs.choice(idx_boot_pool, size=boot_n, replace=True)

        Xb = X[boot_idx]
        yb = y[boot_idx]

        t0 = time.time()
        rf = RandomForestClassifier(**rf_params)
        rf.fit(Xb, yb)

        imp = rf.feature_importances_
        if imp.shape[0] != n_feat:
            raise RuntimeError(f"RF importances length {imp.shape[0]} != n_feat {n_feat}")

        rf_importance_mat[r, :] = imp
        print(f"RF+MDI done in {time.time() - t0:.2f}s")

    rf_mean = pd.Series(rf_importance_mat.mean(axis=0), index=feat_names, name="rf_mean_importance")
    rf_std = pd.Series(rf_importance_mat.std(axis=0), index=feat_names, name="rf_std_importance")
    rf_freq = _topk_freq_from_matrix(rf_importance_mat, feat_names, topk_ref)

    combined_score = _minmax_norm(rf_mean.rename("tmp")).rename("combined_score")
    kept = (rf_freq >= freq_thr).astype(int)

    out_df = pd.DataFrame({
        "feature": feat_names,
        "combined_score": combined_score.values,
        "rf_mean_importance": rf_mean.values,
        "rf_std_importance": rf_std.values,
        "rf_stability_freq": rf_freq.values,
        "kept_by_threshold": kept.values,
    }).sort_values(by=["kept_by_threshold", "combined_score"], ascending=[False, False])

    out_combined.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_combined, index=False)

    print("#############################")
    print(f"Saved combined stability ranking (RF-only): {out_combined}")
    print("Top 20 combined:")
    print(out_df.head(20).to_string(index=False))
    print("#############################")


if __name__ == "__main__":
    main()
