from __future__ import annotations

from datetime import date
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor


def _to_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        v = float(x)
        if np.isnan(v):
            return None
        return v
    except Exception:
        return None


def warmup_response(
    company_id: str,
    reason: str,
    asof_trading_date: Optional[str] = None,
    open_today: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Response used when we cannot produce a valid prediction.
    """
    out: Dict[str, Any] = {
        "company_id": str(company_id),
        "asof_trading_date": asof_trading_date,
        "open_today": open_today,
        "pred_logret_tplus5": None,
        "pred_open_tplus5": (open_today if open_today is not None else None),
        "baseline_open_tplus5": (open_today if open_today is not None else None),
        "warming_up": True,
        "reason": reason,
    }
    return out


def build_feature_vector_from_row(
    row: pd.Series,
    feature_cols: List[str],
) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """
    Extract 18 features from the row in exact order and return shape (1, 18).
    Returns (None, reason) if invalid.
    """
    vals: List[float] = []
    for col in feature_cols:
        if col not in row.index:
            return None, f"Missing feature column in row: {col}"
        v = _to_float(row[col])
        if v is None:
            return None, f"Invalid/NaN feature value for {col}"
        vals.append(v)

    x = np.array(vals, dtype=np.float64).reshape(1, -1)
    return x, None


def predict_from_latest_row(
    model: CatBoostRegressor,
    feature_cols: List[str],
    row: pd.Series,
) -> Dict[str, Any]:
    """
    Predict pred_logret_tplus5 from a latest row (already contains engineered features).
    Convert to pred_open_tplus5 using open_today * exp(pred_logret).
    """
    # asof date
    td = row.get("trading_date")
    asof_str = None
    if isinstance(td, (pd.Timestamp,)):
        asof_str = td.date().isoformat()
    elif td is not None:
        # fallback
        try:
            asof_str = pd.to_datetime(td).date().isoformat()
        except Exception:
            asof_str = None

    open_today = _to_float(row.get("open_price"))
    if open_today is None or open_today <= 0:
        return warmup_response(
            company_id=str(row.get("company_id", row.get("company_id_norm", ""))),
            reason="Invalid open_price in latest row",
            asof_trading_date=asof_str,
            open_today=open_today,
        )

    x, err = build_feature_vector_from_row(row, feature_cols)
    if x is None:
        return warmup_response(
            company_id=str(row.get("company_id", row.get("company_id_norm", ""))),
            reason=err or "Invalid feature vector",
            asof_trading_date=asof_str,
            open_today=open_today,
        )

    pred = model.predict(x)
    # CatBoost can return scalar or array-like
    if isinstance(pred, (list, tuple, np.ndarray)):
        pred_logret = float(np.asarray(pred).reshape(-1)[0])
    else:
        pred_logret = float(pred)

    pred_open_tplus5 = float(open_today * np.exp(pred_logret))

    return {
        "company_id": str(row.get("company_id", row.get("company_id_norm", ""))),
        "asof_trading_date": asof_str,
        "open_today": open_today,
        "pred_logret_tplus5": pred_logret,
        "pred_open_tplus5": pred_open_tplus5,
        "baseline_open_tplus5": open_today,
        "warming_up": False,
        "reason": None,
    }
