"""
Train Stacked Meta-Learner

This script demonstrates the complete workflow for training a stacked ensemble model:
1. Load training and validation data
2. Generate base model predictions (out-of-fold for train, direct for validation)
3. Train meta-learner on base predictions
4. Evaluate and save the stacked model
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from road_accident_risk.modeling.base_models import load_models
from road_accident_risk.modeling.stacked_model import (
    stacked_dataset,
    fit_stacked_model,
    evaluate_stacked_model
)


# =============================================================================
# CONFIGURATION
# =============================================================================

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"


# =============================================================================
# LOAD DATA
# =============================================================================

print("=" * 70)
print("LOADING DATA")
print("=" * 70)

# Load training and validation features
X_train = pd.read_csv(DATA_DIR / "X_train.csv")
X_val = pd.read_csv(DATA_DIR / "X_val.csv")

# Load targets
y_train = pd.read_csv(DATA_DIR / "y_train.csv")
y_val = pd.read_csv(DATA_DIR / "y_val.csv")

print(f"Training set: {X_train.shape}")
print(f"Validation set: {X_val.shape}")
print()


# =============================================================================
# LOAD BASE MODEL PREDICTIONS
# =============================================================================

print("=" * 70)
print("LOADING BASE MODEL PREDICTIONS")
print("=" * 70)

# For training the meta-learner, we need out-of-fold (OOF) predictions
# to avoid overfitting. These should be pre-generated during base model training.

# Load OOF predictions for each base model
oof_lgbm = pd.read_csv(MODEL_DIR / "LGBM" / "oof_preds_lgbm.csv")
oof_xgb = pd.read_csv(MODEL_DIR / "XGB" / "oof_preds_xgb.csv")
oof_tabular = pd.read_csv(MODEL_DIR / "Tabular" / "oof_preds_tabular.csv")

print(f"Loaded LGBM OOF predictions: {oof_lgbm.shape}")
print(f"Loaded XGB OOF predictions: {oof_xgb.shape}")
print(f"Loaded Tabular OOF predictions: {oof_tabular.shape}")
print()

# Load validation predictions for each base model
val_lgbm = pd.read_csv(MODEL_DIR / "LGBM" / "val_preds_lgbm.csv")
val_xgb = pd.read_csv(MODEL_DIR / "XGB" / "val_preds_xgb.csv")
val_tabular = pd.read_csv(MODEL_DIR / "Tabular" / "val_preds_tabular.csv")

print(f"Loaded LGBM validation predictions: {val_lgbm.shape}")
print(f"Loaded XGB validation predictions: {val_xgb.shape}")
print(f"Loaded Tabular validation predictions: {val_tabular.shape}")
print()


# =============================================================================
# TRAIN META-LEARNER
# =============================================================================

print("=" * 70)
print("TRAINING META-LEARNER")
print("=" * 70)

# Combine OOF predictions from all base models
base_learner_train_preds = [oof_lgbm, oof_xgb, oof_tabular]

# Train the meta-learner
stacked_model = fit_stacked_model(base_learner_train_preds, y_train)
print()


# =============================================================================
# EVALUATE ON VALIDATION SET
# =============================================================================

print("=" * 70)
print("EVALUATING ON VALIDATION SET")
print("=" * 70)

# Load the actual base model objects
base_models_dict = load_models()
base_learners = [
    base_models_dict.get("LGBM"),
    base_models_dict.get("XGB"),
    base_models_dict.get("Tabular")
]

# Filter out None values in case some models didn't load
base_learners = [m for m in base_learners if m is not None]

print(f"Using {len(base_learners)} base models for prediction")
print()

# Evaluate the stacked model
rmse, r2 = evaluate_stacked_model(
    base_learners=base_learners,
    meta_model=stacked_model,
    X_test=X_val,
    y_true=y_val,
    save_predictions=True
)


# =============================================================================
# SAVE META-LEARNER
# =============================================================================

print()
print("=" * 70)
print("SAVING META-LEARNER")
print("=" * 70)

# Create output directory
stacked_dir = MODEL_DIR / "Stacked"
stacked_dir.mkdir(parents=True, exist_ok=True)

# Save the trained meta-learner
meta_model_path = stacked_dir / "meta_learner_xgb.pkl"
joblib.dump(stacked_model, meta_model_path)
print(f"Meta-learner saved to: {meta_model_path}")

# Save performance metrics
metrics_df = pd.DataFrame({
    "metric": ["RMSE", "R2"],
    "value": [rmse, r2]
})
metrics_path = stacked_dir / "meta_learner_metrics.csv"
metrics_df.to_csv(metrics_path, index=False)
print(f"Metrics saved to: {metrics_path}")

print()
print("=" * 70)
print("STACKED MODEL TRAINING COMPLETE")
print("=" * 70)

