import pandas as pd
import pytest
from road_accident_risk.data_loader import load_user_input
from road_accident_risk.config import EXPECTED_COLUMNS


def test_load_user_input_complete():
    user = {
        'road_type': 'highway',
        'num_lanes': 4,
        'curvature': 0.02,
        'speed_limit': 100,
        'lighting': 'day',
        'weather': 'clear',
        'road_signs_present': True,
        'public_road': False,
        'time_of_day': 'morning',
        'holiday': False,
        'school_season': True,
        'num_reported_accidents': 2
    }

    df = load_user_input(user)

    # Columns and order
    assert list(df.columns) == list(EXPECTED_COLUMNS.keys())

    # Check dtypes mapping (now using standard dtypes, not nullable)
    assert str(df.dtypes['num_lanes']) == 'int64'
    assert str(df.dtypes['curvature']) == 'float64'
    assert str(df.dtypes['road_signs_present']) == 'bool'

    # Check values preserved
    assert df.at[0, 'road_type'] == 'highway'
    assert int(df.at[0, 'num_lanes']) == 4


def test_load_user_input_missing_fields():
    """Test that missing required fields raise ValueError"""
    incomplete_user = {
        'road_type': 'highway',
        'num_lanes': 4,
        # Missing other required fields
    }

    with pytest.raises(ValueError, match="Missing required fields"):
        load_user_input(incomplete_user)


def test_load_user_input_type_coercion():
    """Test that values are properly coerced to expected types"""
    user = {
        'road_type': 123,  # will be coerced to string
        'num_lanes': '4',  # string to int
        'curvature': '0.05',  # string to float
        'speed_limit': 60,
        'lighting': 'day',
        'weather': 'clear',
        'road_signs_present': 'True',  # string to bool
        'public_road': 1,  # int to bool via string
        'time_of_day': 'morning',
        'holiday': False,
        'school_season': 'yes',  # 'yes' to bool
        'num_reported_accidents': 2.5  # float to int
    }

    df = load_user_input(user)

    # Check coercions
    assert df.at[0, 'road_type'] == '123'
    assert df.at[0, 'num_lanes'] == 4
    assert abs(float(df.at[0, 'curvature']) - 0.05) < 1e-9
    assert bool(df.at[0, 'road_signs_present']) is True
    assert bool(df.at[0, 'school_season']) is True
