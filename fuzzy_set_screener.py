"""
fuzzy_set_screener.py - Re-Engineered Probabilistic Fuzzy Set Screener
======================================================================
Implements rigorous probability foundations from the notes:
  - Quotient Transformation (PDF 1, Page 6): E/P and B/P yields with IQR winsorization
  - Laplace Distribution CDF (PDF 3, Page 4): Double exponential membership for momentum
  - Beta Distribution CDF (PDF 3, Page 3): Bounded variable transformation
  - Bivariate Frank Copula (PDF 1, Pages 1-3): Smooth joint aggregation replacing min(A, B)
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy import stats

from distribution_utils import (
    quotient_earnings_yield,
    quotient_book_yield,
    iqr_winsorize,
    laplace_cdf_membership,
    robust_mad_zscore,
    beta_cdf_membership,
    beta_liquidity_weights,
    copula_aggregate,
)


def run_fuzzy_screener(data_dir: str = r"C:\Users\saral\PycharmProjects\PythonProject1\data",
                       out_csv: str | None = None) -> pd.DataFrame:
    from nepse_fuzzy_screener import load_universe, build_copula_fuzzy_sets

    print(f"[fuzzy_set_screener] Loading universe from {data_dir}...")
    df = load_universe(data_dir=data_dir)
    print(f"[fuzzy_set_screener] Evaluating probabilistic fuzzy sets on {len(df)} assets...")
    scored = build_copula_fuzzy_sets(df)

    out_cols = [
        "symbol", "securityName", "sector", "closePrice",
        "earnings_yield", "book_yield",
        "mu_V", "mu_Q", "mu_HighDebt", "mu_NegEPS", "mu_Risk", "mu_Core",
        "mu_S", "mu_T1", "mu_T2", "mu_M", "mu_Composite", "liquidity_weight",
    ]
    avail_cols = [c for c in out_cols if c in scored.columns]
    ranked = scored[avail_cols].round(4).sort_values("mu_Composite", ascending=False)

    if out_csv is None:
        out_csv = os.path.join(data_dir, "fuzzy_set_screener.csv")

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    ranked.to_csv(out_csv, index=False)
    print(f"[fuzzy_set_screener] Results written to {out_csv}")

    return ranked


if __name__ == "__main__":
    dir_path = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\saral\PycharmProjects\PythonProject1\data"
    ranked = run_fuzzy_screener(data_dir=dir_path)
    print("\n--- TOP 10 STOCKS (Probabilistic Fuzzy Set Screener) ---")
    print(ranked.head(10)[
              ["symbol", "sector", "closePrice", "mu_V", "mu_Q", "mu_Composite", "liquidity_weight"]].to_string(
        index=False))
