import os, json, math
import numpy as np
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

MODEL_PATH = BASE_DIR / os.getenv("MODEL_PATH", "artifacts/catboost_model.cbm")
FEATURE_COLS_PATH = BASE_DIR / os.getenv("FEATURE_COLS_PATH", "artifacts/feature_cols.json")
TEST_DATA_PATH = BASE_DIR / os.getenv("TEST_DATA_PATH", "data/processed_test_tplus5_essential.csv")

def mape(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)

def smape(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean(2*np.abs(y_pred - y_true)/(np.abs(y_true)+np.abs(y_pred))) * 100)

def main():
    feature_cols = json.loads(FEATURE_COLS_PATH.read_text())
    df = pd.read_csv(TEST_DATA_PATH)

    # Load model
    model = CatBoostRegressor()
    model.load_model(str(MODEL_PATH))

    # Predict log-returns
    X = df[feature_cols].values
    pred_logret = np.asarray(model.predict(X)).reshape(-1)

    # Ground truth
    y_logret = df["target_logret_tplus5"].astype(float).values
    open_today = df["open_price"].astype(float).values
    true_open_tplus5 = df["open_tplus5"].astype(float).values

    # Convert to price
    pred_open_tplus5 = open_today * np.exp(pred_logret)

    # Baseline (predict no change)
    baseline_open_tplus5 = open_today

    # Metrics (logret)
    logret_mae = mean_absolute_error(y_logret, pred_logret)
    logret_rmse = math.sqrt(mean_squared_error(y_logret, pred_logret))
    logret_r2 = r2_score(y_logret, pred_logret)
    dir_acc = float(np.mean(np.sign(pred_logret) == np.sign(y_logret)) * 100)
    corr = float(np.corrcoef(pred_logret, y_logret)[0, 1])

    # Metrics (price)
    price_mae = mean_absolute_error(true_open_tplus5, pred_open_tplus5)
    price_rmse = math.sqrt(mean_squared_error(true_open_tplus5, pred_open_tplus5))
    price_mape = mape(true_open_tplus5, pred_open_tplus5)
    price_smape = smape(true_open_tplus5, pred_open_tplus5)
    median_ape = float(np.median(np.abs((true_open_tplus5 - pred_open_tplus5) / true_open_tplus5) * 100))

    # Baseline metrics
    b_mae = mean_absolute_error(true_open_tplus5, baseline_open_tplus5)
    b_rmse = math.sqrt(mean_squared_error(true_open_tplus5, baseline_open_tplus5))
    b_mape = mape(true_open_tplus5, baseline_open_tplus5)

    print("=== LOG-RETURN METRICS (target_logret_tplus5) ===")
    print(f"MAE  : {logret_mae:.6f}")
    print(f"RMSE : {logret_rmse:.6f}")
    print(f"R^2  : {logret_r2:.6f}")
    print(f"Corr : {corr:.6f}")
    print(f"Direction accuracy: {dir_acc:.2f}%")

    print("\n=== PRICE METRICS (open_tplus5) ===")
    print(f"MAE  : {price_mae:.6f}")
    print(f"RMSE : {price_rmse:.6f}")
    print(f"MAPE : {price_mape:.3f}%")
    print(f"sMAPE: {price_smape:.3f}%")
    print(f"Median APE: {median_ape:.3f}%")

    print("\n=== BASELINE (predict open_tplus5 = open_today) ===")
    print(f"MAE  : {b_mae:.6f}")
    print(f"RMSE : {b_rmse:.6f}")
    print(f"MAPE : {b_mape:.3f}%")

    rel_impr = (b_mape - price_mape) / b_mape * 100
    print(f"\nRelative MAPE improvement vs baseline: {rel_impr:.2f}%")

if __name__ == "__main__":
    main()
