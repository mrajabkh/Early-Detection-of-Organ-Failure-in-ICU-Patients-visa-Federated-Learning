# run_gru_sweep.py
# Run GRU experiments with top-K feature limits

from __future__ import annotations
from typing import List, Optional

import pandas as pd
import config
from train_eval_gru import TrainConfig, train_and_eval


def run_sweep(
    ks: List[Optional[int]],
    rank_path: Optional[str] = None,
) -> pd.DataFrame:

    disease = config.DISEASE

    cfg = TrainConfig()

    rows = []

    for k in ks:
        k_name = "all" if k is None else str(int(k))

        print("========================================")
        print(f"GRU run | top_k={k_name}")
        print(f"rank_path={rank_path}")
        print("========================================")

        out = train_and_eval(disease=disease, cfg=cfg, top_k=k, rank_path=rank_path)

        rows.append(
            {
                "top_k": k_name,
                "n_features": out["extra"]["n_features"],
                "runtime_sec": out["extra"]["runtime_sec"],
                "cpu_peak_mib": out["extra"]["cpu_peak_mib"],
                "gpu_peak_mib": out["extra"]["gpu_peak_mib"],
                "train_auroc": out["train"]["auroc"],
                "train_auprc": out["train"]["auprc"],
                "val_auroc": out["val"]["auroc"],
                "val_auprc": out["val"]["auprc"],
                "test_auroc": out["test"]["auroc"],
                "test_auprc": out["test"]["auprc"],
            }
        )

    df = pd.DataFrame(rows)

    out_path = config.gru_results_path(disease)
    df.to_csv(out_path, index=False)

    print("========================================")
    print("Summary:")
    print(df.to_string(index=False))
    print("----------------------------------------")
    print(f"Saved results CSV: {out_path}")
    print("========================================")

    return df


if __name__ == "__main__":
    ks = [20, 40, 60, 80, 100, None]
    rank_path = str(config.stability_combined_path(config.DISEASE))
    run_sweep(ks=ks, rank_path=rank_path)
