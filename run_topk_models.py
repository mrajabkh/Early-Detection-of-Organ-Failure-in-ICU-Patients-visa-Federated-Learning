# run_topk_models.py
# Run models on Top-K selected features and save results.
# Uses memory_profiler for peak RAM (sampled every 100ms).
# Prefers stability_combined ranking when enabled and present.
# Location: Project/Code/run_topk_models.py

from __future__ import annotations

import time
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd

from memory_profiler import memory_usage

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

import config
import preprocess


#############################
# Model factory
#############################
def build_models() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    models_scaled: Dict[str, Any] = {}
    models_imputed: Dict[str, Any] = {}

    if config.MODEL_ENABLED.get("log_reg", False):
        models_scaled["Logistic Regression"] = LogisticRegression(
            max_iter=5000,
            class_weight="balanced",
            n_jobs=-1,
        )

    if config.MODEL_ENABLED.get("knn", False):
        models_scaled["kNN"] = KNeighborsClassifier(
            n_neighbors=15,
            weights="distance",
            n_jobs=-1,
        )

    if config.MODEL_ENABLED.get("svm", False):
        models_scaled["SVM"] = SVC(
            kernel="rbf",
            probability=True,
            class_weight="balanced",
        )

    if config.MODEL_ENABLED.get("decision_tree", False):
        models_imputed["Decision Tree"] = DecisionTreeClassifier(
            max_depth=None,
            random_state=config.SPLIT_RANDOM_STATE,
            class_weight="balanced",
        )

    if config.MODEL_ENABLED.get("random_forest", False):
        params = dict(config.RF_TUNED_PARAMS)
        models_imputed["Random Forest"] = RandomForestClassifier(**params)

    if config.MODEL_ENABLED.get("naive_bayes", False):
        models_imputed["Naive Bayes"] = GaussianNB()

    if config.MODEL_ENABLED.get("xgboost", False):
        models_imputed["XGBoost"] = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            n_jobs=-1,
            random_state=config.SPLIT_RANDOM_STATE,
            tree_method="hist",
        )

    if config.MODEL_ENABLED.get("lightgbm", False):
        params = dict(config.LGBM_TUNED_PARAMS)
        params.pop("objective", None)
        models_imputed["LightGBM"] = LGBMClassifier(**params)

    if config.MODEL_ENABLED.get("catboost", False):
        models_imputed["CatBoost"] = CatBoostClassifier(
            iterations=300,
            learning_rate=0.05,
            depth=6,
            loss_function="Logloss",
            eval_metric="Logloss",
            random_seed=config.SPLIT_RANDOM_STATE,
            verbose=False,
            thread_count=-1,
            class_weights=None,
        )

    return models_scaled, models_imputed


#############################
# Evaluation
#############################
def _fit_predict_metrics(
    model: Any,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, Any]:
    start_time = time.time()

    model.fit(X_train, y_train)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_dec = model.decision_function(X_test)
        y_prob = (y_dec - y_dec.min()) / (y_dec.max() - y_dec.min() + 1e-9)

    y_pred = (y_prob >= 0.5).astype(int)

    end_time = time.time()

    auroc = roc_auc_score(y_test, y_prob)
    auprc = average_precision_score(y_test, y_prob)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    return {
        "AUROC": float(auroc),
        "AUPRC": float(auprc),
        "Accuracy": float(accuracy),
        "Precision": float(precision),
        "Recall": float(recall),
        "F1": float(f1),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "Time_seconds": float(end_time - start_time),
    }


def evaluate_with_peak_ram(
    model: Any,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    interval_sec: float = 0.1,
) -> Dict[str, Any]:
    def _runner():
        return _fit_predict_metrics(model, X_train, X_test, y_train, y_test)

    mem_trace, result = memory_usage(
        (_runner, (), {}),
        interval=interval_sec,
        retval=True,
        max_usage=False,
    )

    peak_mb = float(np.max(mem_trace)) if len(mem_trace) else float("nan")
    result["Peak_RAM_MB"] = peak_mb
    return result


#############################
# Ranking loader
#############################
def load_feature_ranking(valid_feature_names: List[str]) -> List[str]:
    """
    Returns ordered feature list to use for Top-K slicing.
    Prefers stability_combined if enabled and present, else falls back to SHAP file.
    """
    feature_set = set(valid_feature_names)

    if config.USE_STABILITY_RANKING:
        stab_path = config.stability_combined_path(config.DISEASE)
        if stab_path.exists():
            df = pd.read_csv(stab_path)
            if "feature" in df.columns:
                ranked = [f for f in df["feature"].astype(str).tolist() if f in feature_set]
                if ranked:
                    print(f"Using stability_combined ranking: {stab_path}")
                    return ranked

    shap_path = config.shap_path(config.DISEASE)
    df = pd.read_csv(shap_path)
    if "feature" not in df.columns:
        raise ValueError("Ranking file must contain a 'feature' column.")
    ranked = [f for f in df["feature"].astype(str).tolist() if f in feature_set]
    if not ranked:
        raise RuntimeError("No overlap between ranking features and valid features.")
    print(f"Using SHAP ranking: {shap_path}")
    return ranked


#############################
# Main
#############################
def main() -> None:
    features_path = config.features_path(config.DISEASE)
    samples_path = config.samples_path(config.DISEASE)
    out_results = config.results_path(config.DISEASE)

    print("#############################")
    print("Top-K model runs")
    print("#############################")
    print(f"Features: {features_path}")
    print(f"Samples:  {samples_path}")
    print(f"Output:   {out_results}")

    X_df, y, split_data = preprocess.load_build_split(
        features_parquet_path=features_path,
        samples_csv_path=samples_path,
        test_size=config.TEST_SIZE,
        random_state=config.SPLIT_RANDOM_STATE,
        stratify=config.STRATIFY_SPLIT,
        impute_strategy=config.IMPUTE_STRATEGY,
        scale_numeric=True,
    )

    valid_feature_names = split_data.artifacts.valid_feature_names
    feature_index_map = preprocess.make_feature_index_map(valid_feature_names)

    ranked_features = load_feature_ranking(valid_feature_names)

    models_scaled, models_imputed = build_models()

    if models_scaled and (split_data.X_train_scaled is None or split_data.X_test_scaled is None):
        raise RuntimeError("Scaled models enabled but scaled arrays are None. Check preprocess settings.")

    all_results: List[Dict[str, Any]] = []

    for TOP_K in config.TOPK_LIST:
        K = min(int(TOP_K), len(ranked_features))
        top_k_features = ranked_features[:K]
        top_k_indices = [feature_index_map[f] for f in top_k_features]

        X_train_imp_k = split_data.X_train_imputed[:, top_k_indices]
        X_test_imp_k = split_data.X_test_imputed[:, top_k_indices]

        X_train_scl_k = split_data.X_train_scaled[:, top_k_indices]
        X_test_scl_k = split_data.X_test_scaled[:, top_k_indices]

        print("#############################")
        print(f"Top-K = {K}")
        print(f"Train (imp): {X_train_imp_k.shape}, Test (imp): {X_test_imp_k.shape}")
        print(f"Train (scl): {X_train_scl_k.shape}, Test (scl): {X_test_scl_k.shape}")
        print("#############################")

        for name, model in models_scaled.items():
            full_name = f"{name} (top-{K})"
            print(f"Running: {full_name}")
            metrics = evaluate_with_peak_ram(
                model=model,
                X_train=X_train_scl_k,
                X_test=X_test_scl_k,
                y_train=split_data.y_train,
                y_test=split_data.y_test,
                interval_sec=0.1,
            )
            all_results.append({
                "Model": full_name,
                "TopK": K,
                **metrics,
            })

        for name, model in models_imputed.items():
            full_name = f"{name} (top-{K})"
            print(f"Running: {full_name}")
            metrics = evaluate_with_peak_ram(
                model=model,
                X_train=X_train_imp_k,
                X_test=X_test_imp_k,
                y_train=split_data.y_train,
                y_test=split_data.y_test,
                interval_sec=0.1,
            )
            all_results.append({
                "Model": full_name,
                "TopK": K,
                **metrics,
            })

    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values(by=["TopK", "AUROC"], ascending=[True, False])

    # Round numeric metrics to 3 d.p. (keep confusion matrix counts as ints)
    float_cols = results_df.select_dtypes(include=[np.number]).columns.tolist()
    int_like_cols = {"TopK", "TN", "FP", "FN", "TP"}
    round_cols = [c for c in float_cols if c not in int_like_cols]
    for c in round_cols:
        results_df[c] = results_df[c].astype(float).round(3)

    for c in ["TopK", "TN", "FP", "FN", "TP"]:
        if c in results_df.columns:
            results_df[c] = results_df[c].astype(int)

    out_results.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(out_results, index=False, float_format="%.3f")

    print("#############################")
    print(f"Saved results to: {out_results}")
    print("#############################")
    print(results_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
