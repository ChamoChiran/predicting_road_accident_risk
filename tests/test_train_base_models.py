import pandas as pd
import pytest

from road_accident_risk.modeling.base_models import train_base_models


def _mock_trainer_xgb(X, y, random_state, params):
    # simple mock: return a tuple to indicate it was called with shapes
    return ("xgb_mock", X.shape, y.shape, random_state, params)


def _mock_trainer_lgbm(X, y, random_state, params):
    return ("lgbm_mock", X.shape, y.shape, random_state, params)


def test_train_base_models_with_mock_trainers():
    # small synthetic dataset without nulls
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4, 5, 6]})
    y = pd.Series([0.1, 0.2, 0.3])

    trainers = {"xgb": _mock_trainer_xgb, "lgbm": _mock_trainer_lgbm}
    models = train_base_models(X, y, learners=("xgb", "lgbm"), trainers=trainers)

    assert "xgb" in models and "lgbm" in models
    assert models["xgb"][0] == "xgb_mock"
    assert models["lgbm"][0] == "lgbm_mock"


def test_train_base_models_raises_on_nulls():
    X = pd.DataFrame({"a": [1.0, None, 3.0], "b": [4, 5, 6]})
    y = pd.Series([0.1, 0.2, 0.3])

    with pytest.raises(ValueError):
        train_base_models(X, y, learners=("xgb",), trainers={"xgb": _mock_trainer_xgb})

