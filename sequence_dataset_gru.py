# sequence_dataset_gru.py
# Build patient-level sequences from rolling-window features.parquet + samples.csv
# Outputs (X, y, length) per patient for GRU/LSTM models.

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

import config


@dataclass
class SplitIndices:
    train_pids: np.ndarray
    val_pids: np.ndarray
    test_pids: np.ndarray


def _read_samples(samples_path: str) -> pd.DataFrame:
    df = pd.read_csv(samples_path)
    need = {"patientunitstayid", "t_end", "label"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"samples.csv missing columns: {sorted(missing)}")

    df = df[["patientunitstayid", "t_end", "label"]].copy()

    df = df.replace([np.inf, -np.inf], np.nan)
    df["patientunitstayid"] = pd.to_numeric(df["patientunitstayid"], errors="coerce")
    df["t_end"] = pd.to_numeric(df["t_end"], errors="coerce")
    df["label"] = pd.to_numeric(df["label"], errors="coerce")

    df = df.dropna(subset=["patientunitstayid", "t_end", "label"])

    df["patientunitstayid"] = df["patientunitstayid"].astype(np.int64)
    df["t_end"] = df["t_end"].astype(np.int64)
    df["label"] = df["label"].astype(np.int64)

    return df



def _read_features(features_path: str) -> pd.DataFrame:
    df = pd.read_parquet(features_path)
    need = {"patientunitstayid", "t_end"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"features.parquet missing columns: {sorted(missing)}")

    df = df.copy()

    # Robust numeric coercion (handles NaN/inf/object types)
    df["patientunitstayid"] = pd.to_numeric(df["patientunitstayid"], errors="coerce")
    df["t_end"] = pd.to_numeric(df["t_end"], errors="coerce")

    # Drop invalid rows before casting to int
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["patientunitstayid", "t_end"])

    # Cast after cleanup
    df["patientunitstayid"] = df["patientunitstayid"].astype(np.int64)
    df["t_end"] = df["t_end"].astype(np.int64)

    return df


def _select_feature_columns(df: pd.DataFrame) -> List[str]:
    # Keep numeric columns only, drop id/time/label.
    drop = {"patientunitstayid", "t_end", "label"}
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cols = [c for c in numeric_cols if c not in drop]
    if not cols:
        raise ValueError("No numeric feature columns found after filtering.")
    return cols


def make_patient_splits(
    pids: np.ndarray,
    seed: int = 42,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
) -> SplitIndices:
    if val_frac <= 0 or test_frac <= 0 or (val_frac + test_frac) >= 0.9:
        raise ValueError("Bad split fractions. Use something like val=0.15, test=0.15.")

    pids = np.unique(pids.astype(np.int64))
    rng = np.random.default_rng(seed)
    rng.shuffle(pids)

    n = len(pids)
    n_test = int(round(n * test_frac))
    n_val = int(round(n * val_frac))

    test_pids = pids[:n_test]
    val_pids = pids[n_test : n_test + n_val]
    train_pids = pids[n_test + n_val :]

    return SplitIndices(train_pids=train_pids, val_pids=val_pids, test_pids=test_pids)


def _compute_norm_stats(df: pd.DataFrame, feature_cols: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    x = df[feature_cols].to_numpy(dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def _apply_norm(df: pd.DataFrame, feature_cols: List[str], mean: np.ndarray, std: np.ndarray) -> None:
    x = df[feature_cols].to_numpy(dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x = (x - mean[None, :]) / std[None, :]
    df.loc[:, feature_cols] = x


def _restrict_to_topk_features(
    df: pd.DataFrame,
    feature_cols: List[str],
    top_k: int,
    rank_path: Optional[str],
) -> List[str]:
    top_k = int(top_k)
    if top_k <= 0:
        raise ValueError("top_k must be a positive integer")

    ranked: Optional[List[str]] = None

    if rank_path is not None:
        r = pd.read_csv(rank_path)
        if "feature" not in r.columns:
            raise ValueError("rank_path must be a CSV with a 'feature' column")
        ranked = [f for f in r["feature"].astype(str).tolist() if f in feature_cols]

    if ranked is None or len(ranked) == 0:
        # Fallback: rank by variance on the current df
        tmp = df[feature_cols].to_numpy(dtype=np.float32)
        tmp = np.nan_to_num(tmp, nan=0.0, posinf=0.0, neginf=0.0)
        var = tmp.var(axis=0)
        order = np.argsort(-var)
        ranked = [feature_cols[i] for i in order.tolist()]

    return ranked[:top_k]


class PatientSequenceDataset(Dataset):
    """
    Returns per-patient sequences:
      X: (T, D) float32
      y: (T,) int64
      length: int64 (<= max_len)
    """

    def __init__(
        self,
        split: str,
        disease: Optional[str] = None,
        max_len: int = 128,
        seed: int = 42,
        val_frac: float = 0.15,
        test_frac: float = 0.15,
        normalize: bool = True,
        cached_norm_path: Optional[str] = None,
        top_k: Optional[int] = None,
        rank_path: Optional[str] = None,
    ) -> None:
        super().__init__()
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be one of: train, val, test")

        if disease is None:
            disease = getattr(config, "DISEASE", None)
        if disease is None:
            raise ValueError("disease is None and config.DISEASE not found.")

        self.split = split
        self.max_len = int(max_len)

        samples_path = str(config.samples_path(disease))
        features_path = str(config.features_path(disease))

        samples = _read_samples(samples_path)
        feats = _read_features(features_path)

        df = samples.merge(feats, on=["patientunitstayid", "t_end"], how="inner")
        if df.empty:
            raise ValueError("Merged dataset is empty. Check that samples.csv and features.parquet align.")

        df = df.sort_values(["patientunitstayid", "t_end"]).reset_index(drop=True)

        feature_cols = _select_feature_columns(df)

        # Patient splits
        splits = make_patient_splits(
            pids=df["patientunitstayid"].to_numpy(),
            seed=seed,
            val_frac=val_frac,
            test_frac=test_frac,
        )

        if split == "train":
            keep = set(splits.train_pids.tolist())
        elif split == "val":
            keep = set(splits.val_pids.tolist())
        else:
            keep = set(splits.test_pids.tolist())

        df = df[df["patientunitstayid"].isin(keep)].copy()
        df = df.sort_values(["patientunitstayid", "t_end"]).reset_index(drop=True)
        if df.empty:
            raise ValueError(f"No rows left after applying split='{split}'. Check split fractions/seed.")

        # Optional: restrict to top-K features
        if top_k is not None:
            feature_cols = _restrict_to_topk_features(df, feature_cols, top_k=top_k, rank_path=rank_path)

        self.feature_cols = feature_cols

        # Normalization (train stats only)
        self.mean = None
        self.std = None
        if normalize:
            if cached_norm_path is not None:
                norm = np.load(cached_norm_path)
                self.mean = norm["mean"].astype(np.float32)
                self.std = norm["std"].astype(np.float32)
            else:
                # Compute on TRAIN split from the full merged dataset (then apply to this split)
                df_train = samples.merge(feats, on=["patientunitstayid", "t_end"], how="inner")
                df_train = df_train.sort_values(["patientunitstayid", "t_end"]).reset_index(drop=True)
                df_train = df_train[df_train["patientunitstayid"].isin(set(splits.train_pids.tolist()))].copy()
                df_train = df_train.sort_values(["patientunitstayid", "t_end"]).reset_index(drop=True)

                # If we restricted to top_k, compute stats only for those same columns
                mean, std = _compute_norm_stats(df_train, self.feature_cols)
                self.mean, self.std = mean, std

            _apply_norm(df, self.feature_cols, self.mean, self.std)

        # Store
        self.df = df
        self.pids = df["patientunitstayid"].unique().astype(np.int64)

        # Pre-store group indices for speed
        self._groups: List[np.ndarray] = []
        pid_values = df["patientunitstayid"].to_numpy(dtype=np.int64)

        start = 0
        for pid in self.pids:
            while start < len(pid_values) and pid_values[start] != pid:
                start += 1
            end = start
            while end < len(pid_values) and pid_values[end] == pid:
                end += 1
            idx = np.arange(start, end, dtype=np.int64)
            self._groups.append(idx)
            start = end

    def __len__(self) -> int:
        return len(self.pids)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        idx = self._groups[i]
        sub = self.df.iloc[idx]

        x = sub[self.feature_cols].to_numpy(dtype=np.float32)  # (T, D)
        y = sub["label"].to_numpy(dtype=np.int64)  # (T,)

        # Truncate to max_len from the end (most recent windows)
        if len(y) > self.max_len:
            x = x[-self.max_len :]
            y = y[-self.max_len :]

        length = np.int64(len(y))

        return (
            torch.from_numpy(x),  # (T, D)
            torch.from_numpy(y),  # (T,)
            torch.tensor(length, dtype=torch.long),
        )


def pad_collate(batch):
    """
    batch: list of (X (Ti,D), y (Ti,), length)
    returns:
      X_pad: (B, T_max, D)
      y_pad: (B, T_max)
      mask:  (B, T_max) 1 for valid
      lengths: (B,)
    """
    xs, ys, lens = zip(*batch)
    lengths = torch.stack(lens, dim=0)  # (B,)

    bsz = len(xs)
    d = xs[0].shape[1]
    t_max = int(lengths.max().item())

    x_pad = torch.zeros((bsz, t_max, d), dtype=torch.float32)
    y_pad = torch.zeros((bsz, t_max), dtype=torch.long)
    mask = torch.zeros((bsz, t_max), dtype=torch.float32)

    for i in range(bsz):
        t = int(lengths[i].item())
        x_pad[i, :t, :] = xs[i]
        y_pad[i, :t] = ys[i]
        mask[i, :t] = 1.0

    return x_pad, y_pad, mask, lengths
