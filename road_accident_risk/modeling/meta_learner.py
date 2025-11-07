"""
STAGE 2: META-LEARNER (STACKED ENSEMBLE)
=========================================

This script combines predictions from multiple base models (Stage 1) to produce
a single, more accurate final prediction.

"""

import numpy as np
import pandas as pd
import joblib
import sys
import streamlit as st

# Conditional imports based on execution context
if __name__ == "__main__":
    # When running as a script, add parent directory to path and use absolute imports
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from road_accident_risk.config import MODEL_PATHS, STACKED_MODEL_DIR, ROOT
    from road_accident_risk.data_loader import load_user_input
    from road_accident_risk.modeling.base_models import get_base_predictions
else:
    # When imported as a module, use relative imports
    from ..config import ROOT, STACKED_MODEL_DIR
    from .base_models import get_base_predictions
    from ..data_loader import load_user_input


@st.cache_resource
def load_meta_model():
    if STACKED_MODEL_DIR.exists():
        model = joblib.load(STACKED_MODEL_DIR / "stacked_model.pkl")
        print(f"✓ Loaded meta-model from {STACKED_MODEL_DIR / 'meta_model.pkl'}")
        return model
    else:
        raise FileNotFoundError(f"Meta model directory not found: {STACKED_MODEL_DIR}")


def predict_from_stage1_output(base_predictions: pd.DataFrame) -> np.ndarray:
    """
    Generate final predictions from base model outputs.

    Args:
        base_predictions: DataFrame with predictions from each base model.

    Returns:
        Final stacked prediction.
    """
    print("Stage 2: meta model prediction from stage 1 output")

    # load meta model and predict
    print("Loading meta model and generating final prediction")
    meta_model = load_meta_model()

    # Extract prediction columns
    pred_cols = [c for c in base_predictions.columns if c.endswith('_preds')]
    base_predictions_arr = base_predictions[pred_cols].values

    # Generate final prediction
    final_preds = meta_model.predict(base_predictions_arr)
    print(f"Generated final predictions: {final_preds.shape}")

    return final_preds


def predict_stacked(X: pd.DataFrame, y_true=None) -> pd.DataFrame:
    print("Stage 2: meta model prediction")

    print("Generating base model predictions...")
    base_predictions = get_base_predictions(X)
    print(f"Base predictions shape: {base_predictions.shape}")

    # generate final prediction
    final_preds = predict_from_stage1_output(base_predictions)

    # combine results
    results_df = base_predictions.copy()
    results_df['stacked_prediction'] = final_preds

    return results_df



if __name__ == "__main__":
    # Example: Complete workflow with user input
    # Example user input
    sample_input = {
        "road_type": "highway",
        "num_lanes": 2,
        "curvature": 0.5,
        "speed_limit": 60,
        "lighting": "night",
        "weather": "clear",
        "road_signs_present": False,
        "public_road": False,
        "time_of_day": "afternoon",
        "holiday": True,
        "school_season": True,
        "num_reported_accidents": 0,
    }
    sample_input = pd.DataFrame([sample_input])  # Wrap in list to create single row

    # STAGE 1: Get base model predictions
    stage1_output = get_base_predictions(sample_input)

    # STAGE 2: Get stacked model predictions
    print("Generating stacked model predictions...")
    stacked_output = predict_stacked(sample_input)

    print("\n" + "=" * 60)
    print("STAGE 1 OUTPUT (input for Stage 2):")
    print("=" * 60)
    print(stacked_output)
    print("\n")
