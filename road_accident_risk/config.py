"""Configuration for the road_accident_risk package.

Holds shared constants such as the expected training schema.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

EXPECTED_COLUMNS = {
    "road_type": "object",
    "num_lanes": "int64",
    "curvature": "float64",
    "speed_limit": "int64",
    "lighting": "object",
    "weather": "object",
    "road_signs_present": "bool",
    "public_road": "bool",
    "time_of_day": "object",
    "holiday": "bool",
    "school_season": "bool",
    "num_reported_accidents": "int64",
}

MODEL_PATHS = {
    "LGBM": ROOT / "models" / "LGBM" / "reg_lgbm.pkl",
    "XGB": ROOT / "models" / "XGB" / "reg_xgb.pkl",
    "Tabular": ROOT / "models" / "Tabular" / "tabular_model.pkl",
}

STACKED_MODEL_DIR = ROOT / "models" / "Stacked"


