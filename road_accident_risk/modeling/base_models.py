"""
STAGE 1: BASE MODEL PREDICTIONS
================================
This script loads pre-trained models and generates predictions that will be
used as input for the meta model (Stage 2).

Flow:
    User Input → load_user_input() → get_base_predictions() → Stage 2 Meta Model
"""

from pathlib import Path
import joblib
import torch
import pandas as pd
import sys
import streamlit as st

# Conditional imports based on execution context
if __name__ == "__main__":
    from road_accident_risk.config import MODEL_PATHS
    from road_accident_risk.data_loader import load_user_input
else:
    from ..config import MODEL_PATHS
    from ..data_loader import load_user_input


# Load models
@st.cache_resource
def load_models():
    """
    Load all available pre-trained models from disk.
    Returns a dictionary with model name as key and model object as value.
    """
    models = {}

    # Load lightgbm model
    lgbm_path = Path(MODEL_PATHS["LGBM"])
    if lgbm_path.exists():
        models["LGBM"] = joblib.load(lgbm_path)
        print(f"Loaded LGBM model from {lgbm_path}")

    # Load xgboost model
    xgb_path = Path(MODEL_PATHS["XGB"])
    if xgb_path.exists():
        models["XGB"] = joblib.load(xgb_path)
        print(f"Loaded XGB model from {xgb_path}")

    # Load PyTorch Tabular model
    tabular_path = Path(MODEL_PATHS["Tabular"])
    if tabular_path.exists():
        models['Tabular'] = torch.load(
            tabular_path,
            map_location=torch.device("cpu")
        )
        print(f"Loaded Tabular model from {tabular_path}")

    print(f"Successfully loaded {len(models)} models.")

    return models


# get stage-1 outputs
def get_base_predictions(df_input: pd.DataFrame, include_input=False):
    """
    Generate predictions from all available base models.

    Args:
        df_input: DataFrame with input features (same columns as training data)
        include_input: Whether to include input features in output

    Returns:
        DataFrame with predictions from each model
    """

    # Load all available models
    models = load_models()
    predictions = {}


    for model_name, model in models.items():
        print(f"Predicting with {model_name}...")

        if model_name in ["LGBM", "XGB"]:
            predictions[f"{model_name.lower()}_preds"] = model.predict(df_input)

        elif model_name == "Tabular":
            df_temp = df_input.copy()

            # convert categorical columns to string type
            categorical_columns = ["road_type", "lighting", "weather", "time_of_day"]
            for col in categorical_columns:
                if col in df_temp.columns:
                    df_temp[col] = df_temp[col].astype(str)

            # get predictions
            output = model.predict(df_temp)
            predictions[f"{model_name.lower()}_preds"] = output["accident_risk_prediction"].values

        else:
            print(f"Unknown model: {model_name}")

    # Combine predictions into a DataFrame
    df_predictions = pd.DataFrame(predictions)

    print(f"Successfully generated {len(predictions)} predictions.")
    return df_predictions


# Main function
def predict_from_user_input(user_dict: dict, include_input=False):
    """
    Main function to get predictions from user input.

    Takes a dictionary with user input, validates it, converts to proper format,
    and returns predictions from all base models.

    Args:
        user_dict: Dictionary with all required input fields
        include_input: Whether to include input features in output

    Returns:
        DataFrame with predictions from each model
    """
    df_formatted = load_user_input(user_dict)
    print(f"Successfully loaded {len(df_formatted)} user inputs.")

    # get predictions from all the models
    print(f"getting predictions form models...")
    df_predictions = get_base_predictions(df_formatted, include_input=include_input)

    print(f"Successfully generated {len(df_predictions)} predictions.")
    return df_predictions



if __name__ == "__main__":
    """
    Example: How to use this script for Stage 1 predictions
    """

    # When running as a script, add parent directory to path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

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

    # STAGE 1: Get base model predictions
    stage1_output = predict_from_user_input(sample_input)

    print("\n" + "=" * 60)
    print("STAGE 1 OUTPUT (input for Stage 2):")
    print("=" * 60)
    print(stage1_output)
    print("\n")
else:
    # When imported as a module, use relative imports
    from ..config import MODEL_PATHS
    from ..data_loader import load_user_input
