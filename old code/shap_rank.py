# shap_rank.py
# Compute SHAP global feature importance using tuned LightGBM.
# Location: Project/Code/shap_rank.py

from __future__ import annotations

import numpy as np
import pandas as pd
import shap

from lightgbm import LGBMClassifier

import config
import preprocess


#############################
# Main
#############################
def main() -> None:
    features_path = config.features_path(config.DISEASE)
    samples_path = config.samples_path(config.DISEASE)
    out_shap = config.shap_path(config.DISEASE)

    print("#############################")
    print("SHAP ranking (LightGBM)")
    print("#############################")
    print(f"Features: {features_path}")
    print(f"Samples:  {samples_path}")
    print(f"Output:   {out_shap}")

    # Load + preprocess
    X_df, y, split_data = preprocess.load_build_split(
        features_parquet_path=features_path,
        samples_csv_path=samples_path,
        test_size=config.TEST_SIZE,
        random_state=config.SPLIT_RANDOM_STATE,
        stratify=config.STRATIFY_SPLIT,
        impute_strategy=config.IMPUTE_STRATEGY,
        scale_numeric=False,  # SHAP + LGBM use imputed, no scaling needed
    )

    valid_feature_names = split_data.artifacts.valid_feature_names

    # Rebuild imputed training DataFrame with correct names
    X_train_imp_df = pd.DataFrame(
        split_data.X_train_imputed,
        columns=valid_feature_names,
    )

    print("#############################")
    print(f"Valid features after train-only drop-all-NaN: {len(valid_feature_names)}")
    print(f"X_train_imp_df shape: {X_train_imp_df.shape}")
    print("#############################")

    # Fit tuned LGBM
    params = dict(config.LGBM_TUNED_PARAMS)
    # Remove params LGBMClassifier may not accept in some versions (safe guard)
    params.pop("objective", None)

    lgbm = LGBMClassifier(**params)
    lgbm.fit(X_train_imp_df, split_data.y_train)

    # Background sample for SHAP
    if len(X_train_imp_df) > config.SHAP_BACKGROUND_N:
        background = X_train_imp_df.sample(n=config.SHAP_BACKGROUND_N, random_state=config.SEED)
    else:
        background = X_train_imp_df

    print("#############################")
    print(f"SHAP background shape: {background.shape}")
    print("#############################")

    explainer = shap.TreeExplainer(lgbm)
    shap_values = explainer.shap_values(background)

    # shap_values can be list [class0, class1] or array depending on shap version
    if isinstance(shap_values, list):
        shap_vals_pos = shap_values[1]
    else:
        shap_vals_pos = shap_values

    mean_abs_shap = np.abs(shap_vals_pos).mean(axis=0)

    shap_df = pd.DataFrame({
        "feature": valid_feature_names,
        "mean_abs_shap": mean_abs_shap,
    }).sort_values("mean_abs_shap", ascending=False)

    print("Top 20 SHAP features:")
    print(shap_df.head(20).to_string(index=False))

    out_shap.parent.mkdir(parents=True, exist_ok=True)
    shap_df.to_csv(out_shap, index=False)
    print("#############################")
    print(f"Saved SHAP ranking to: {out_shap}")
    print("#############################")


if __name__ == "__main__":
    main()
