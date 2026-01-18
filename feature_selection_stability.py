# feature_selection_stability.py
# Multi-model stability feature selection:
# - LightGBM + SHAP (primary)
# - Random Forest importance (secondary; fast MDI by default)
# Combines rankings into a single stability_combined CSV.
# Location: Project/Code/feature_selection_stability.py

from __future__ import annotations

import time
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd

import shap

from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

import config
import preprocess


#############################
# Utilities
#############################
def set_seeds(seed: int) -> None:
    np.random.seed(seed)


def _rank_series_desc(values: pd.Series) -> pd.Series:
    return values.rank(ascending=False, method="average")


def _topk_freq_from_matrix(importance_mat: np.ndarray, feature_names: List[str], topk_ref: int) -> pd.Series:
    n_runs, n_feat = importance_mat.shape
    K = min(int(topk_ref), n_feat)

    counts = np.zeros(n_feat, dtype=np.int32)
    for i in range(n_runs):
        idx = np.argsort(-importance_mat[i])[:K]
        counts[idx] += 1

    return pd.Series(counts / float(n_runs), index=feature_names, name="stability_freq")


def _minmax_norm(s: pd.Series) -> pd.Series:
    mn = float(s.min())
    mx = float(s.max())
    if mx <= mn:
        return pd.Series(np.zeros(len(s), dtype=np.float64), index=s.index)
    return (s - mn) / (mx - mn)


#############################
# Main
#############################
def main() -> None:
    set_seeds(config.SEED)

    features_path = config.features_path(config.DISEASE)
    samples_path = config.samples_path(config.DISEASE)

    out_lgbm = config.stability_lgbm_path(config.DISEASE)
    out_rf = config.stability_rf_path(config.DISEASE)
    out_combined = config.stability_combined_path(config.DISEASE)

    print("#############################")
    print("Stability feature selection")
    print("#############################")
    print(f"Features: {features_path}")
    print(f"Samples:  {samples_path}")
    print(f"Out LGBM: {out_lgbm}")
    print(f"Out RF:   {out_rf}")
    print(f"Out Comb: {out_combined}")

    X_df, y, split_data = preprocess.load_build_split(
        features_parquet_path=features_path,
        samples_csv_path=samples_path,
        test_size=config.TEST_SIZE,
        random_state=config.SPLIT_RANDOM_STATE,
        stratify=config.STRATIFY_SPLIT,
        impute_strategy=config.IMPUTE_STRATEGY,
        scale_numeric=True,
    )

    feat_names = split_data.artifacts.valid_feature_names
    n_feat = len(feat_names)

    X_train_imp = split_data.X_train_imputed
    X_train_scl = split_data.X_train_scaled
    y_train = split_data.y_train

    if X_train_scl is None:
        raise RuntimeError("Expected scaled arrays for stability selection, got None.")

    idx_all = np.arange(len(y_train), dtype=np.int32)
    idx_boot_pool, idx_val = train_test_split(
        idx_all,
        test_size=0.25,
        random_state=config.SEED,
        stratify=y_train if config.STRATIFY_SPLIT else None,
    )

    X_val_imp = X_train_imp[idx_val]
    y_val = y_train[idx_val]

    # Optional cap for speed (only used if permutation mode is enabled)
    if hasattr(config, "PERM_MAX_SAMPLES") and config.PERM_MAX_SAMPLES is not None:
        if len(y_val) > int(config.PERM_MAX_SAMPLES):
            rs = np.random.RandomState(config.SEED)
            sub_idx = rs.choice(len(y_val), size=int(config.PERM_MAX_SAMPLES), replace=False)
            X_val_imp = X_val_imp[sub_idx]
            y_val = y_val[sub_idx]

    print("#############################")
    print(f"Train rows: {len(y_train)}")
    print(f"Boot pool rows: {len(idx_boot_pool)}")
    print(f"Val rows (perm if used): {len(y_val)}")
    print(f"Features: {n_feat}")
    print("#############################")

    n_runs = int(config.STAB_N_BOOTSTRAPS)
    boot_frac = float(config.STAB_BOOTSTRAP_FRAC)

    lgbm_importance_mat = np.zeros((n_runs, n_feat), dtype=np.float64)
    rf_importance_mat = np.zeros((n_runs, n_feat), dtype=np.float64)

    lgbm_params = dict(config.LGBM_TUNED_PARAMS)
    lgbm_params.pop("objective", None)

    rf_params = dict(config.RF_TUNED_PARAMS)

    rf_mode = getattr(config, "RF_IMPORTANCE_MODE", "mdi").lower()
    print(f"RF importance mode: {rf_mode}")

    for r in range(n_runs):
        print("#############################")
        print(f"Bootstrap {r + 1}/{n_runs}")
        print("#############################")

        rs = np.random.RandomState(config.SEED + 1000 + r)

        boot_n = max(10, int(round(boot_frac * len(idx_boot_pool))))
        boot_idx = rs.choice(idx_boot_pool, size=boot_n, replace=True)

        Xb_imp = X_train_imp[boot_idx]
        yb = y_train[boot_idx]

        #############################
        # Primary: LightGBM + SHAP
        #############################
        t0 = time.time()

        lgbm = LGBMClassifier(**lgbm_params)
        Xb_imp_df = pd.DataFrame(Xb_imp, columns=feat_names)
        lgbm.fit(Xb_imp_df, yb)

        if len(Xb_imp_df) > int(config.STAB_SHAP_BACKGROUND_N):
            bg = Xb_imp_df.sample(n=int(config.STAB_SHAP_BACKGROUND_N), random_state=config.SEED + r)
        else:
            bg = Xb_imp_df

        explainer = shap.TreeExplainer(lgbm)
        shap_values = explainer.shap_values(bg)

        if isinstance(shap_values, list):
            shap_vals_pos = shap_values[1]
        else:
            shap_vals_pos = shap_values

        mean_abs = np.abs(shap_vals_pos).mean(axis=0)
        lgbm_importance_mat[r, :] = mean_abs

        print(f"LGBM+SHAP done in {time.time() - t0:.2f}s")

        #############################
        # Secondary: Random Forest importance
        #############################
        t1 = time.time()

        rf = RandomForestClassifier(**rf_params)
        rf.fit(Xb_imp, yb)

        if rf_mode == "mdi":
            rf_importance_mat[r, :] = rf.feature_importances_
            print(f"RF+MDI done in {time.time() - t1:.2f}s")
        elif rf_mode == "permutation":
            # Optional: permutation only on top-M features by MDI (much faster)
            max_feats = getattr(config, "PERM_MAX_FEATURES", 200)
            max_feats = int(max_feats)

            mdi = rf.feature_importances_
            top_idx = np.argsort(-mdi)[:min(max_feats, len(mdi))]

            perm = permutation_importance(
                rf,
                X_val_imp[:, top_idx],
                y_val,
                scoring=getattr(config, "PERM_SCORING", "average_precision"),
                n_repeats=int(getattr(config, "PERM_N_REPEATS", 1)),
                random_state=config.SEED + r,
                n_jobs=-1,
            )

            # Fill only the evaluated indices, others stay 0
            tmp = np.zeros(n_feat, dtype=np.float64)
            tmp[top_idx] = perm.importances_mean
            rf_importance_mat[r, :] = tmp

            print(f"RF+perm(top-{len(top_idx)}) done in {time.time() - t1:.2f}s")
        else:
            raise ValueError(f"Unknown RF_IMPORTANCE_MODE: {rf_mode}")

    #############################
    # Summarize stability per selector
    #############################
    lgbm_mean = pd.Series(lgbm_importance_mat.mean(axis=0), index=feat_names, name="mean_importance")
    lgbm_std = pd.Series(lgbm_importance_mat.std(axis=0), index=feat_names, name="std_importance")
    lgbm_freq = _topk_freq_from_matrix(lgbm_importance_mat, feat_names, config.STAB_TOPK_REF)

    rf_mean = pd.Series(rf_importance_mat.mean(axis=0), index=feat_names, name="mean_importance")
    rf_std = pd.Series(rf_importance_mat.std(axis=0), index=feat_names, name="std_importance")
    rf_freq = _topk_freq_from_matrix(rf_importance_mat, feat_names, config.STAB_TOPK_REF)

    lgbm_df = pd.concat([lgbm_mean, lgbm_std, lgbm_freq], axis=1).reset_index().rename(columns={"index": "feature"})
    rf_df = pd.concat([rf_mean, rf_std, rf_freq], axis=1).reset_index().rename(columns={"index": "feature"})

    lgbm_df = lgbm_df.sort_values(by=["stability_freq", "mean_importance"], ascending=[False, False])
    rf_df = rf_df.sort_values(by=["stability_freq", "mean_importance"], ascending=[False, False])

    out_lgbm.parent.mkdir(parents=True, exist_ok=True)
    lgbm_df.to_csv(out_lgbm, index=False)
    rf_df.to_csv(out_rf, index=False)

    print("#############################")
    print(f"Saved LGBM stability: {out_lgbm}")
    print(f"Saved RF stability:   {out_rf}")
    print("#############################")

    #############################
    # Combine selectors
    #############################
    lgbm_rank = _rank_series_desc(lgbm_mean)
    rf_rank = _rank_series_desc(rf_mean)

    lgbm_rank_norm = _minmax_norm((lgbm_rank.max() + 1.0) - lgbm_rank)
    rf_rank_norm = _minmax_norm((rf_rank.max() + 1.0) - rf_rank)

    combined_score = 0.5 * lgbm_rank_norm + 0.5 * rf_rank_norm

    keep_mask = (lgbm_freq >= float(config.STAB_FREQ_THRESHOLD)) | (rf_freq >= float(config.STAB_FREQ_THRESHOLD))

    combined_df = pd.DataFrame({
        "feature": feat_names,
        "combined_score": combined_score.values,
        "lgbm_mean_shap": lgbm_mean.values,
        "lgbm_stability_freq": lgbm_freq.values,
        "rf_mean_importance": rf_mean.values,
        "rf_stability_freq": rf_freq.values,
        "kept_by_threshold": keep_mask.values.astype(int),
    })

    combined_df = combined_df.sort_values(
        by=["kept_by_threshold", "combined_score"],
        ascending=[False, False],
    )

    combined_df.to_csv(out_combined, index=False)

    print("#############################")
    print(f"Saved combined stability ranking: {out_combined}")
    print("Top 20 combined:")
    print(combined_df.head(20).to_string(index=False))
    print("#############################")


if __name__ == "__main__":
    main()
