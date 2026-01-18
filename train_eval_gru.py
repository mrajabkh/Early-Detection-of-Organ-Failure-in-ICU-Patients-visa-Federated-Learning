# train_eval_gru.py
# Train + evaluate a supervised GRU early detection model on window-sequence data.

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import config
from sequence_dataset_gru import PatientSequenceDataset, pad_collate
from gru_risk import GRURisk

try:
    from memory_profiler import memory_usage
except Exception:
    memory_usage = None


# -------------------------
# Metrics
# -------------------------
def roc_auc_score_manual(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = y_true.astype(np.int64)
    y_score = y_score.astype(np.float64)

    pos = y_true == 1
    neg = y_true == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(y_score)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(len(y_score)) + 1

    sorted_scores = y_score[order]
    sorted_pos = pos[order]

    i = 0
    sum_ranks_pos = 0.0
    while i < len(sorted_scores):
        j = i
        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        avg_rank = 0.5 * (ranks[order[i]] + ranks[order[j]])
        sum_ranks_pos += int(sorted_pos[i : j + 1].sum()) * avg_rank
        i = j + 1

    return (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def avg_precision_score_manual(y_true: np.ndarray, y_score: np.ndarray) -> float:
    n_pos = int((y_true == 1).sum())
    if n_pos == 0:
        return float("nan")

    order = np.argsort(-y_score)
    y_sorted = y_true[order]

    tp = 0
    fp = 0
    ap = 0.0
    prev_rec = 0.0

    for i in range(len(y_sorted)):
        if y_sorted[i] == 1:
            tp += 1
        else:
            fp += 1
        rec = tp / n_pos
        prec = tp / max(tp + fp, 1)
        if rec > prev_rec:
            ap += prec * (rec - prev_rec)
            prev_rec = rec

    return ap


def flatten_valid(logits, y, mask):
    probs = torch.sigmoid(logits)
    valid = mask > 0.5
    return (
        y[valid].detach().cpu().numpy().astype(np.int64),
        probs[valid].detach().cpu().numpy().astype(np.float64),
    )


def masked_bce(logits, targets, mask, pos_weight=None):
    loss_fn = nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_weight)
    loss = loss_fn(logits, targets.float())
    loss = loss * mask.float()
    return loss.sum() / mask.sum().clamp_min(1.0)


# -------------------------
# Config
# -------------------------
@dataclass
class TrainConfig:
    max_len: int = 128
    batch_size: int = 32
    hidden_dim: int = 128
    num_layers: int = 1
    dropout: float = 0.2
    lr: float = 1e-3
    weight_decay: float = 1e-5
    epochs: int = 20
    patience: int = 3
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# -------------------------
# Eval
# -------------------------
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_y, all_s = [], []
    total_loss = 0.0

    for x, y, mask, lengths in loader:
        x = x.to(device)
        y = y.to(device)
        mask = mask.to(device)

        logits = model(x, lengths)
        loss = masked_bce(logits, y, mask)
        total_loss += float(loss.item())

        yt, ys = flatten_valid(logits, y, mask)
        if len(yt):
            all_y.append(yt)
            all_s.append(ys)

    if not all_y:
        return {"loss": float("nan"), "auroc": float("nan"), "auprc": float("nan")}

    y_true = np.concatenate(all_y)
    y_score = np.concatenate(all_s)

    return {
        "loss": total_loss / max(len(loader), 1),
        "auroc": roc_auc_score_manual(y_true, y_score),
        "auprc": avg_precision_score_manual(y_true, y_score),
    }


# -------------------------
# Train + Eval
# -------------------------
def train_and_eval(
    disease: config.DiseaseSpec,
    cfg: TrainConfig,
    top_k: int | None = None,
    rank_path: str | None = None,
) -> Dict[str, Dict]:

    t0 = time.perf_counter()

    if cfg.device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    def _run():
        train_ds = PatientSequenceDataset(
            split="train",
            disease=disease,
            max_len=cfg.max_len,
            seed=cfg.seed,
            normalize=True,
            top_k=top_k,
            rank_path=rank_path,
        )
        val_ds = PatientSequenceDataset(
            split="val",
            disease=disease,
            max_len=cfg.max_len,
            seed=cfg.seed,
            normalize=True,
            top_k=top_k,
            rank_path=rank_path,
        )
        test_ds = PatientSequenceDataset(
            split="test",
            disease=disease,
            max_len=cfg.max_len,
            seed=cfg.seed,
            normalize=True,
            top_k=top_k,
            rank_path=rank_path,
        )

        train_loader = DataLoader(train_ds, cfg.batch_size, True, collate_fn=pad_collate)
        val_loader = DataLoader(val_ds, cfg.batch_size, False, collate_fn=pad_collate)
        test_loader = DataLoader(test_ds, cfg.batch_size, False, collate_fn=pad_collate)

        model = GRURisk(
            input_dim=len(train_ds.feature_cols),
            hidden_dim=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
        ).to(cfg.device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

        best_val = -1.0
        best_state = None
        bad_epochs = 0

        for epoch in range(cfg.epochs):
            model.train()
            for x, y, mask, lengths in train_loader:
                x = x.to(cfg.device)
                y = y.to(cfg.device)
                mask = mask.to(cfg.device)

                optimizer.zero_grad()
                loss = masked_bce(model(x, lengths), y, mask)
                loss.backward()
                optimizer.step()

            val = evaluate(model, val_loader, cfg.device)
            if val["auroc"] > best_val:
                best_val = val["auroc"]
                best_state = model.state_dict()
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= cfg.patience:
                    break

        model.load_state_dict(best_state)

        return {
            "train": evaluate(model, train_loader, cfg.device),
            "val": evaluate(model, val_loader, cfg.device),
            "test": evaluate(model, test_loader, cfg.device),
            "n_features": len(train_ds.feature_cols),
        }

    if memory_usage is not None:
        mem, out = memory_usage((_run, (), {}), retval=True, interval=0.1)
        cpu_peak = float(max(mem))
    else:
        out = _run()
        cpu_peak = float("nan")

    runtime = time.perf_counter() - t0
    gpu_peak = (
        torch.cuda.max_memory_allocated() / (1024 ** 2)
        if cfg.device.startswith("cuda") and torch.cuda.is_available()
        else float("nan")
    )

    out["extra"] = {
        "runtime_sec": runtime,
        "cpu_peak_mib": cpu_peak,
        "gpu_peak_mib": gpu_peak,
        "n_features": out["n_features"],
    }

    return out
