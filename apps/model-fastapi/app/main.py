from __future__ import annotations

import os
from datetime import date
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query

from app.data_source_csv import get_latest_row_for_company, load_test_dataframe
from app.model_loader import load_catboost_model, load_feature_cols
from app.predictor import predict_from_latest_row, warmup_response

# Resolve base dir = apps/model-fastapi/
BASE_DIR = Path(__file__).resolve().parents[1]

# Load environment variables from .env
load_dotenv(dotenv_path=BASE_DIR / ".env")

MODEL_PATH = os.getenv("MODEL_PATH", "artifacts/catboost_model.cbm")
FEATURE_COLS_PATH = os.getenv("FEATURE_COLS_PATH", "artifacts/feature_cols.json")
TEST_DATA_PATH = os.getenv("TEST_DATA_PATH", "data/processed_test_tplus5_essential.csv")
DATA_SOURCE = os.getenv("DATA_SOURCE", "csv")
N_LOOKBACK = int(os.getenv("N_LOOKBACK", "30"))  # kept for later DB mode, not required for this CSV

app = FastAPI(title="Open Price Prediction Model Service", version="0.1.0")


def _resolve(p: str) -> Path:
    path = Path(p)
    return (BASE_DIR / path).resolve() if not path.is_absolute() else path.resolve()


@app.on_event("startup")
def startup_load() -> None:
    # Load feature columns + model
    feature_cols_path = _resolve(FEATURE_COLS_PATH)
    model_path = _resolve(MODEL_PATH)

    feature_cols = load_feature_cols(feature_cols_path)
    model = load_catboost_model(model_path)

    app.state.feature_cols = feature_cols
    app.state.model = model

    # Load CSV test dataset into memory (Stage A)
    if DATA_SOURCE.lower() == "csv":
        csv_path = _resolve(TEST_DATA_PATH)
        bundle = load_test_dataframe(csv_path)
        app.state.df = bundle.df
        app.state.df_rows = bundle.n_rows
    else:
        # DB mode later
        app.state.df = None
        app.state.df_rows = 0


@app.get("/health")
def health():
    ok_model = hasattr(app.state, "model")
    ok_cols = hasattr(app.state, "feature_cols")
    df_rows = getattr(app.state, "df_rows", 0)
    return {
        "status": "ok",
        "data_source": DATA_SOURCE,
        "model_loaded": bool(ok_model),
        "feature_cols_loaded": bool(ok_cols),
        "feature_cols_count": (len(app.state.feature_cols) if ok_cols else 0),
        "csv_rows_loaded": df_rows,
    }


@app.get("/predict")
def predict(
    company_id: str = Query(..., description="Company identifier (ticker-like in the test CSV, e.g., AAF)"),
    asof: Optional[date] = Query(None, description="Optional as-of date (YYYY-MM-DD) to predict using latest row <= asof"),
):
    # Only CSV mode for now
    if DATA_SOURCE.lower() != "csv":
        raise HTTPException(status_code=501, detail="DATA_SOURCE != csv not implemented yet. Use csv for Stage A.")

    df = getattr(app.state, "df", None)
    if df is None:
        raise HTTPException(status_code=500, detail="CSV dataset not loaded.")

    row = get_latest_row_for_company(df, company_id=company_id, asof=asof)
    if row is None:
        return warmup_response(
            company_id=company_id,
            reason="company_id not found in test dataset (or no rows <= asof)",
            asof_trading_date=(asof.isoformat() if asof else None),
            open_today=None,
        )

    model = app.state.model
    feature_cols = app.state.feature_cols
    return predict_from_latest_row(model=model, feature_cols=feature_cols, row=row)
