from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd


@dataclass(frozen=True)
class CsvDataBundle:
    df: pd.DataFrame
    n_rows: int


def load_test_dataframe(csv_path: Path) -> CsvDataBundle:
    """
    Load the processed test dataset into memory once.
    Ensures:
      - trading_date parsed
      - sorted by company_id, trading_date
      - company_id normalized to string (company_id_norm)
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Test CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    required = {"company_id", "trading_date", "open_price"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {sorted(missing)}")

    # Parse date and normalize company_id for matching
    df["trading_date"] = pd.to_datetime(df["trading_date"], errors="coerce")
    df = df.dropna(subset=["trading_date"])
    df["company_id_norm"] = df["company_id"].astype(str)

    # Sort for stable "latest row" selection
    df = df.sort_values(["company_id_norm", "trading_date"]).reset_index(drop=True)

    return CsvDataBundle(df=df, n_rows=len(df))


def get_latest_row_for_company(
    df: pd.DataFrame,
    company_id: str,
    asof: Optional[date] = None,
) -> Optional[pd.Series]:
    """
    Return the latest row for a given company_id (ticker-like string in your test file).
    If asof is provided, pick the latest row with trading_date <= asof.
    """
    cid = str(company_id)

    sub = df[df["company_id_norm"] == cid]
    if sub.empty:
        return None

    if asof is not None:
        asof_dt = pd.to_datetime(asof)
        sub = sub[sub["trading_date"] <= asof_dt]
        if sub.empty:
            return None

    # latest = last row after sort
    return sub.iloc[-1]
