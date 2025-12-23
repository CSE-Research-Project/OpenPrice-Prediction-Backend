from __future__ import annotations

import json
from pathlib import Path
from typing import List

from catboost import CatBoostRegressor


def load_feature_cols(feature_cols_path: Path) -> List[str]:
    """
    Load feature column order from feature_cols.json.
    Must be a JSON array of strings with length 18.
    """
    raw = feature_cols_path.read_text(encoding="utf-8")
    obj = json.loads(raw)

    if not isinstance(obj, list) or not all(isinstance(x, str) for x in obj):
        raise ValueError("feature_cols.json must be a JSON array of strings.")

    if len(obj) != 18:
        raise ValueError(f"feature_cols.json must contain 18 features, got {len(obj)}.")

    return obj


def load_catboost_model(model_path: Path) -> CatBoostRegressor:
    """
    Load CatBoost model from .cbm file.
    """
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = CatBoostRegressor()
    model.load_model(str(model_path))
    return model
