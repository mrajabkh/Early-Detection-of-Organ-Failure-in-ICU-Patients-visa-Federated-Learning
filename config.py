# config.py
# Central config for the rolling-window ICU prediction pipeline.
# Location: Project/Code/config.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict, Any


#############################
# Project layout (auto-detected)
#############################
CODE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CODE_DIR.parent

EICU_DATA_DIR = PROJECT_ROOT / "eICU(v2.0)"
OUTPUTS_DIR = PROJECT_ROOT / "Outputs"


#############################
# Task definition (disease label)
#############################
@dataclass(frozen=True)
class DiseaseSpec:
    major: str
    subcategory: Optional[str] = None


DISEASE = DiseaseSpec(
    major="cardiovascular",
    subcategory="cardiac arrest",
)


#############################
# Rolling window settings
#############################
HISTORY_HRS = 12
HORIZON_HRS = 12
STRIDE_MINS = 60

HISTORY_MINS = HISTORY_HRS * 60
HORIZON_MINS = HORIZON_HRS * 60


#############################
# Windowing strategy (Strategy 1)
#############################
WINDOWING_STRATEGY = "strategy1"
NON_EVENT_N_MULT = 2  # N = 2H


#############################
# Optional: cap total windows by sampling patients
#############################
WINDOW_CAP_ENABLED = True
MAX_WINDOWS = 1_000_000
WINDOW_CAP_MODE = "neg_only"  # "both" or "neg_only"
WINDOW_CAP_RANDOM_STATE = 42


#############################
# Sample generation
#############################
SEED = 42
REQUIRE_FULL_HORIZON = True


#############################
# Optional: balancing (global)
#############################
# This controls the 1:1 behavior you observed.
# If BALANCE_ENABLED is False, prepare_data will NOT downsample negatives to match positives.
#bad dont use
BALANCE_ENABLED = False

# Only used if BALANCE_ENABLED is True.
NEGATIVE_POSITIVE_RATIO = 1.0
NEGATIVE_WINDOWS_PER_PATIENT_CAP: Optional[int] = None


#############################
# Train/val/test split
#############################
TEST_SIZE = 0.2
VAL_SIZE = 0.15
SPLIT_RANDOM_STATE = 42
STRATIFY_SPLIT = True


#############################
# Test-only monitoring density filter (Option 3)
#############################
APPLY_TEST_DENSITY_FILTER = True
TEST_DENSITY_POS_QUANTILE = 0.25

# Vitals definition for density: any presence in these columns counts as monitored
VP_VITAL_COLS = [
    "heartrate",
    "respiration",
    "sao2",
    "temperature",
    "systemicmean",
    "systemicsystolic",
    "systemicdiastolic",
]

VA_VITAL_COLS = [
    "noninvasivemean",
    "noninvasivesystolic",
    "noninvasivediastolic",
]


#############################
# Negative limiter (cap Neg:Pos) applied PER SPLIT after split assignment
#############################
# NOTE: Name kept for compatibility, but it now applies to train/val/test.
TEST_NEG_LIMITER_ENABLED = True
TEST_NEG_POS_MAX_RATIO = 10.0
TEST_NEG_LIMITER_RANDOM_STATE = 42


USE_POS_WEIGHT = True
POS_WEIGHT_MAX = 500.0

FS_MAX_TRAIN_ROWS = 70000
FS_STRATIFIED_SUBSAMPLE = True
FS_RANDOM_STATE = 42

# how many parquet columns to read per batch (smaller = less RAM)
FS_PARQUET_COL_BATCH = 40


#############################
# Preprocessing (baseline ML)
#############################
IMPUTE_STRATEGY = "median"
SCALE_NUMERIC = True


#############################
# Feature selection (Top-K)
#############################
TOPK_LIST: List[int] = [20, 40, 60, 80, 100]


#############################
# Respiratory and nursing charting feature extraction
#############################
RESPCHART_TOP_LABELS = 40
NURSECHART_TOP_LABELS = 48


#############################
# Stability selection settings (RF-only now)
#############################
USE_STABILITY_RANKING = True

STAB_N_BOOTSTRAPS = 15
STAB_BOOTSTRAP_FRAC = 0.7
STAB_TOPK_REF = 100
STAB_FREQ_THRESHOLD = 0.6

RF_IMPORTANCE_MODE = "mdi"

PERM_N_REPEATS = 1
PERM_SCORING = "average_precision"
PERM_MAX_SAMPLES = 2000
PERM_MAX_FEATURES = 200


#############################
# Model toggles (baseline ML)
#############################
MODEL_ENABLED = {
    "knn": True,
    "random_forest": True,
    "lightgbm": True,
    "log_reg": False,
    "svm": False,
    "decision_tree": False,
    "naive_bayes": False,
    "xgboost": False,
    "catboost": False,
}


#############################
# Tuned model params (locked in)
#############################
LGBM_TUNED_PARAMS: Dict[str, Any] = {
    "subsample": 1.0,
    "num_leaves": 63,
    "n_estimators": 200,
    "min_child_samples": 20,
    "max_depth": 20,
    "learning_rate": 0.1,
    "colsample_bytree": 0.7,
    "objective": "binary",
    "class_weight": "balanced",
    "random_state": 42,
    "n_jobs": -1,
    "verbosity": -1,
}

RF_TUNED_PARAMS: Dict[str, Any] = {
    "n_estimators": 500,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "max_features": 0.5,
    "max_depth": None,
    "random_state": 42,
    "n_jobs": -1,
    "class_weight": "balanced",
}


#############################
# Output naming helpers
#############################
def slugify(s: str) -> str:
    out = s.strip().lower()
    out = out.replace("/", "_").replace("\\", "_")
    out = out.replace("|", "_")
    out = out.replace(" ", "_")
    while "__" in out:
        out = out.replace("__", "_")
    return out


def disease_tag(disease: DiseaseSpec = DISEASE) -> str:
    major = slugify(disease.major)
    sub = slugify(disease.subcategory) if disease.subcategory else "all"
    return f"{major}__{sub}"


def run_name(disease: DiseaseSpec = DISEASE) -> str:
    return (
        f"{disease_tag(disease)}"
        f"__hist{HISTORY_HRS}h"
        f"__hor{HORIZON_HRS}h"
        f"__stride{STRIDE_MINS}m"
    )


def run_dir(disease: DiseaseSpec = DISEASE) -> Path:
    d = OUTPUTS_DIR / run_name(disease)
    d.mkdir(parents=True, exist_ok=True)
    return d


#############################
# Output filenames
#############################
def samples_filename(disease: DiseaseSpec = DISEASE) -> str:
    return f"samples__{disease_tag(disease)}.csv"


def features_filename(disease: DiseaseSpec = DISEASE) -> str:
    return f"features__{disease_tag(disease)}.parquet"


def results_filename(disease: DiseaseSpec = DISEASE) -> str:
    return f"ML_results__{disease_tag(disease)}.csv"


def stability_rf_filename(disease: DiseaseSpec = DISEASE) -> str:
    return f"stability_rf_perm__{disease_tag(disease)}.csv"


def stability_combined_filename(disease: DiseaseSpec = DISEASE) -> str:
    return f"stability_combined__{disease_tag(disease)}.csv"


def gru_results_filename(disease: DiseaseSpec = DISEASE) -> str:
    return f"gru_results__{disease_tag(disease)}.csv"


def gru_checkpoint_filename(disease: DiseaseSpec = DISEASE) -> str:
    return f"gru_checkpoint__{disease_tag(disease)}.pt"


#############################
# Output paths
#############################
def samples_path(disease: DiseaseSpec = DISEASE) -> Path:
    return run_dir(disease) / samples_filename(disease)


def features_path(disease: DiseaseSpec = DISEASE) -> Path:
    return run_dir(disease) / features_filename(disease)


def results_path(disease: DiseaseSpec = DISEASE) -> Path:
    return run_dir(disease) / results_filename(disease)


def stability_rf_path(disease: DiseaseSpec = DISEASE) -> Path:
    return run_dir(disease) / stability_rf_filename(disease)


def stability_combined_path(disease: DiseaseSpec = DISEASE) -> Path:
    return run_dir(disease) / stability_combined_filename(disease)


def gru_results_path(disease: DiseaseSpec = DISEASE) -> Path:
    return run_dir(disease) / gru_results_filename(disease)


def gru_checkpoint_path(disease: DiseaseSpec = DISEASE) -> Path:
    return run_dir(disease) / gru_checkpoint_filename(disease)
