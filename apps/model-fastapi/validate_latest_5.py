import os
import numpy as np
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

from app.model_loader import load_catboost_model, load_feature_cols
from app.data_source_csv import load_test_dataframe, get_latest_row_for_company
from app.predictor import predict_from_latest_row

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

MODEL_PATH = BASE_DIR / os.getenv("MODEL_PATH", "artifacts/catboost_model.cbm")
FEATURE_COLS_PATH = BASE_DIR / os.getenv("FEATURE_COLS_PATH", "artifacts/feature_cols.json")
TEST_DATA_PATH = BASE_DIR / os.getenv("TEST_DATA_PATH", "data/processed_test_tplus5_essential.csv")

def main():
    feature_cols = load_feature_cols(FEATURE_COLS_PATH)
    model = load_catboost_model(MODEL_PATH)
    bundle = load_test_dataframe(TEST_DATA_PATH)
    df = bundle.df

    # pick 5 companies with enough rows (more stable)
    counts = df["company_id_norm"].value_counts()
    candidates = counts[counts >= 30].index.tolist()
    if len(candidates) < 5:
        candidates = df["company_id_norm"].unique().tolist()

    np.random.seed(42)
    sample_ids = list(np.random.choice(candidates, size=min(5, len(candidates)), replace=False))

    results = []
    for cid in sample_ids:
        row = get_latest_row_for_company(df, company_id=cid)
        pred = predict_from_latest_row(model=model, feature_cols=feature_cols, row=row)

        # Ground truth (exists in your processed test set)
        true_logret = float(row.get("target_logret_tplus5")) if "target_logret_tplus5" in row.index else None
        true_open_tplus5 = float(row.get("open_tplus5")) if "open_tplus5" in row.index else None

        pred_open = pred["pred_open_tplus5"]
        pred_logret = pred["pred_logret_tplus5"]
        open_today = pred["open_today"]
        asof = pred["asof_trading_date"]

        abs_err = (pred_open - true_open_tplus5) if (true_open_tplus5 is not None) else None
        pct_err = (abs_err / true_open_tplus5 * 100.0) if (abs_err is not None and true_open_tplus5 != 0) else None

        # Direction correctness: sign of predicted vs true log-return
        dir_ok = None
        if true_logret is not None and pred_logret is not None:
            dir_ok = (np.sign(pred_logret) == np.sign(true_logret))

        results.append({
            "company_id": cid,
            "asof": asof,
            "open_today": open_today,
            "pred_open_tplus5": pred_open,
            "true_open_tplus5": true_open_tplus5,
            "abs_err": abs_err,
            "pct_err_%": pct_err,
            "pred_logret": pred_logret,
            "true_logret": true_logret,
            "direction_ok": dir_ok,
        })

    out = pd.DataFrame(results)
    # prettier display
    pd.set_option("display.width", 180)
    pd.set_option("display.max_columns", 50)
    print(out.to_string(index=False))

    # quick summary metrics (on rows with ground truth)
    valid = out.dropna(subset=["true_open_tplus5", "pred_open_tplus5"])
    if not valid.empty:
        mae = (valid["pred_open_tplus5"] - valid["true_open_tplus5"]).abs().mean()
        rmse = np.sqrt(((valid["pred_open_tplus5"] - valid["true_open_tplus5"]) ** 2).mean())
        mape = (valid["pred_open_tplus5"] - valid["true_open_tplus5"]).abs().div(valid["true_open_tplus5"]).mean() * 100.0
        print("\nSummary (these 5 latest points only):")
        print(f"MAE  = {mae:.4f}")
        print(f"RMSE = {rmse:.4f}")
        print(f"MAPE = {mape:.2f}%")

if __name__ == "__main__":
    main()
