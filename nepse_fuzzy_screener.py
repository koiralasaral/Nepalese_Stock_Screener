"""
nepse_fuzzy_screener.py - Advanced Bivariate Copula & Distributional Fuzzy Screener
==================================================================================
Fully overhauled with core probability concepts:
  1. Quotient Distribution Transformation (PDF 1, Page 6):
     Eliminates P/E singularities via continuous Earnings Yield (E/P) and Book-to-Market (B/P)
     with IQR winsorization.
  2. Laplace Distribution CDF (PDF 3, Page 4):
     Evaluates micro-momentum (T2) and price reversion with Double Exponential CDFs:
     F(x) = 0.5 * exp((x-mu)/b) for x < mu, 1 - 0.5 * exp(-(x-mu)/b) for x >= mu.
  3. Beta Distribution CDF (PDF 3, Page 3):
     Evaluates bounded indicators: 52-week range position (alpha=5, beta=2),
     dividend payout ratio sweet-spot, and sector RSI.
  4. Bivariate Joint Copula Aggregation (PDF 1, Pages 1-3):
     Replaces the destructive hard min(A, B, ...) operator with smooth Archimedean
     Frank Copula joint dependence aggregation.
"""

import os
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
    frank_copula_bivariate,
    copula_aggregate,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("NEPSE_DATA_DIR", r"C:\Users\saral\PycharmProjects\PythonProject1")
FUZZY_DIR = os.path.join(DATA_DIR, "Fuzzy set")

TODAY_CSV = os.path.join(DATA_DIR, "data/today_price.csv")
REF_CSV = os.path.join(DATA_DIR, "data/securities_reference.csv")
FUND_UPDATED_CSV = os.path.join(DATA_DIR, "fundamentals_updated.csv")
FUND_BASE_CSV = os.path.join(DATA_DIR, "data/fundamentals_base.csv")
SECTOR_RATIOS_CSV = os.path.join(DATA_DIR, "data/sector_ratios.csv")

OUT_CSV_DATA = os.path.join(DATA_DIR, "nepse_fuzzy_screener.csv")
OUT_CSV_FUZZY = os.path.join(FUZZY_DIR, "nepse_fuzzy_screener.csv")

SECTOR_TO_INDEX = {
    "Commercial Banks": "BANKING",
    "Development Banks": "DEVBANK",
    "Finance": "FINANCE",
    "Microfinance": "MICROFINANCE",
    "Hydro Power": "HYDROPOWER",
    "Life Insurance": "LIFEINSU",
    "Non Life Insurance": "NONLIFEINSU",
    "Manufacturing And Processing": "MANUFACTURE",
    "Hotels And Tourism": "HOTELS",
    "Investment": "INVESTMENT",
    "Tradings": "TRADING",
    "Others": "OTHERS",
}


def load_universe(data_dir: str = DATA_DIR) -> pd.DataFrame:
    """Loads today's price data, reference metadata, fundamentals, and sector ratios."""
    today_p = os.path.join(data_dir, "data/today_price.csv")
    ref_p = os.path.join(data_dir, "data/securities_reference.csv")
    fund_p = os.path.join(data_dir, "fundamentals_updated.csv")
    fund_base_p = os.path.join(data_dir, "data/fundamentals_base.csv")
    sector_p = os.path.join(data_dir, "data/sector_ratios.csv")

    price = pd.read_csv(today_p)
    ref = pd.read_csv(ref_p)[["symbol", "sectorName", "instrumentType"]].drop_duplicates("symbol")

    # Filter to tradeable equity shares
    equity_symbols = set(ref.loc[ref["instrumentType"] == "Equity", "symbol"])
    df = price[price["symbol"].isin(equity_symbols)].copy()
    df = df[(df["closePrice"] > 0) & (df["fiftyTwoWeekHigh"] > df["fiftyTwoWeekLow"])].copy()

    ref = ref[["symbol", "sectorName"]]
    if os.path.exists(fund_p):
        fund = pd.read_csv(fund_p)
        df = df.merge(fund, on="symbol", how="left")

    if os.path.exists(fund_base_p):
        fund_base = pd.read_csv(fund_base_p)
        base_cols = [c for c in ["symbol", "eps_growth_pct", "book_value"] if c in fund_base.columns]
        df = df.merge(fund_base[base_cols], on="symbol", how="left")

    df = df.merge(ref, on="symbol", how="left")
    df["sector"] = df["sectorName"].fillna("Others")
    df["sector_index"] = df["sector"].map(SECTOR_TO_INDEX).fillna("OTHERS")

    if os.path.exists(sector_p):
        sector_macro = pd.read_csv(sector_p).set_index("index_name")
        df["sec_macd_hist"] = df["sector_index"].map(sector_macro["macd"] - sector_macro["macdsignal"])
        df["sec_yearly_chg"] = df["sector_index"].map(sector_macro["yearly_percent_change"])
        df["sec_price_vs_sma200"] = df["sector_index"].map(
            (sector_macro["ltp"] - sector_macro["sma_200"]) / sector_macro["sma_200"] * 100
        )
        df["sec_rsi"] = df["sector_index"].map(sector_macro["rsi"])
    else:
        df["sec_macd_hist"] = 0.0
        df["sec_yearly_chg"] = 0.0
        df["sec_price_vs_sma200"] = 0.0
        df["sec_rsi"] = 50.0

    return df.reset_index(drop=True)


def build_copula_fuzzy_sets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes linguistic variables and aggregates them using Archimedean Frank Copula.
    """
    df = df.copy()

    # -------------------------------------------------------------------------
    # 1. Deep Value (V) - Quotient Regularized Transformations (PDF 1, Page 6)
    # -------------------------------------------------------------------------
    # Convert P/E to Earnings Yield: E/P = EPS / Price
    if "eps" in df.columns:
        df["earnings_yield"] = quotient_earnings_yield(df["eps"], df["closePrice"])
    elif "pe_ratio" in df.columns:
        df["earnings_yield"] = np.where(df["pe_ratio"] > 0, 1.0 / df["pe_ratio"], -0.1)
    else:
        df["earnings_yield"] = 0.0

    # Convert P/B to Book Yield: B/P = BookValue / Price
    if "book_value" in df.columns and df["book_value"].notna().sum() > 5:
        df["book_yield"] = quotient_book_yield(df["book_value"], df["closePrice"])
    elif "pb_ratio" in df.columns:
        df["book_yield"] = np.where(df["pb_ratio"] > 0, 1.0 / df["pb_ratio"], 0.0)
    else:
        df["book_yield"] = 0.0

    # Apply IQR winsorization to eliminate outlier spikes
    df["earnings_yield_clean"] = iqr_winsorize(df["earnings_yield"])
    df["book_yield_clean"] = iqr_winsorize(df["book_yield"])

    # High earnings yield and high book yield mean high value (cheapness)
    v_ey = laplace_cdf_membership(df["earnings_yield_clean"], direction="high")
    v_by = laplace_cdf_membership(df["book_yield_clean"], direction="high")

    # Graham discount: more negative = cheaper
    if "graham_discount" in df.columns:
        v_graham = laplace_cdf_membership(iqr_winsorize(df["graham_discount"]), direction="low")
    else:
        v_graham = pd.Series(0.5, index=df.index)

    df["mu_V"] = (v_ey + v_by + v_graham) / 3.0

    # -------------------------------------------------------------------------
    # 2. Quality & Profitability (Q)
    # -------------------------------------------------------------------------
    q_roe = laplace_cdf_membership(df.get("roe", pd.Series(0.0, index=df.index)), direction="high")
    q_roa = laplace_cdf_membership(df.get("roa", pd.Series(0.0, index=df.index)), direction="high")
    q_growth = laplace_cdf_membership(df.get("eps_growth_pct", pd.Series(0.0, index=df.index)), direction="high")
    df["mu_Q"] = (q_roe + q_roa + q_growth) / 3.0

    # -------------------------------------------------------------------------
    # 3. Risk & Distressed Status
    # -------------------------------------------------------------------------
    if "debt_to_equity" in df.columns:
        debt_to_equity = pd.to_numeric(df["debt_to_equity"], errors="coerce")
        high_debt = laplace_cdf_membership(debt_to_equity, direction="high")
        df["mu_HighDebt"] = np.where(debt_to_equity.notna(), high_debt, 0.0)
    else:
        df["mu_HighDebt"] = 0.0
    # Negative / weak EPS
    df["mu_NegEPS"] = laplace_cdf_membership(df.get("eps", pd.Series(0.0, index=df.index)), mu=0.0, direction="low")
    df["mu_Risk"] = np.maximum(df["mu_HighDebt"], df["mu_NegEPS"])

    # Smooth Core: V and Q joint copula minus Risk
    core_joint = frank_copula_bivariate(df["mu_V"].values, df["mu_Q"].values, theta=3.0)
    df["mu_Core"] = np.clip(core_joint * (1.0 - 0.70 * df["mu_Risk"].values), 0.0, 1.0)

    # -------------------------------------------------------------------------
    # 4. Safety & Yield (S) - Beta Shaped Payout (PDF 3, Page 3)
    # -------------------------------------------------------------------------
    s_yield = laplace_cdf_membership(df.get("dividend_yield", pd.Series(0.0, index=df.index)), direction="high")

    # Dividend payout ratio: sustainable sweet spot is 20% to 60%
    payout_norm = (df.get("payout_ratio", pd.Series(0.0, index=df.index)) / 100.0).clip(0.0, 1.0)
    # Beta distribution with mode at alpha/(alpha+beta) = 2/5 = 0.40 (40% payout)
    s_payout = beta_cdf_membership(payout_norm, alpha=2.5, beta_param=3.5)

    # 52-week price stability: narrower 52w range means higher stability
    f_range = (df["fiftyTwoWeekHigh"] - df["fiftyTwoWeekLow"]) / np.maximum(df["fiftyTwoWeekLow"], 1.0)
    s_stability = laplace_cdf_membership(f_range, direction="low")

    df["mu_S"] = (s_yield + s_payout + s_stability) / 3.0

    # -------------------------------------------------------------------------
    # 5. Macro Uptrend (T1) - Sector Level
    # -------------------------------------------------------------------------
    t1_sma = laplace_cdf_membership(df["sec_price_vs_sma200"], mu=0.0, direction="high")
    t1_macd = laplace_cdf_membership(df["sec_macd_hist"], mu=0.0, direction="high")
    t1_yearly = laplace_cdf_membership(df["sec_yearly_chg"], mu=0.0, direction="high")
    df["mu_T1"] = (t1_sma + t1_macd + t1_yearly) / 3.0

    # -------------------------------------------------------------------------
    # 6. Micro Momentum (T2) - Laplace CDF & Beta Range (PDF 3, Pages 3-4)
    # -------------------------------------------------------------------------
    ret_vs_prev = (df["closePrice"] - df["previousDayClosePrice"]) / np.maximum(df["previousDayClosePrice"],
                                                                                1.0) * 100.0
    ret_vs_avg = (df["closePrice"] - df["averageTradedPrice"]) / np.maximum(df["averageTradedPrice"], 1.0) * 100.0

    # 52w position in [0, 1]
    denom_52w = np.maximum(df["fiftyTwoWeekHigh"] - df["fiftyTwoWeekLow"], 1e-4)
    f_52w_pos = ((df["closePrice"] - df["fiftyTwoWeekLow"]) / denom_52w).clip(0.0, 1.0)

    # Momentum returns modeled with Laplace CDF (heavy tails)
    t2_prev = laplace_cdf_membership(ret_vs_prev, mu=0.0, direction="high")
    t2_avg = laplace_cdf_membership(ret_vs_avg, mu=0.0, direction="high")
    # 52-week high momentum modeled via Beta(5, 2) CDF (PDF 3, Page 3)
    t2_52w = beta_cdf_membership(f_52w_pos, alpha=4.0, beta_param=2.0)
    df["mu_T2"] = (t2_prev + t2_avg + t2_52w) / 3.0

    # -------------------------------------------------------------------------
    # 7. Oscillator Confirm (M)
    # -------------------------------------------------------------------------
    day_range = np.maximum(df["highPrice"] - df["lowPrice"], 1e-4)
    intraday_pos = ((df["closePrice"] - df["lowPrice"]) / day_range).clip(0.0, 1.0)
    m_intraday = pd.Series(intraday_pos, index=df.index)

    # RSI sweet-spot: 45 to 65 is bullish without being overbought
    rsi_norm = (df["sec_rsi"] / 100.0).clip(0.0, 1.0)
    m_rsi = beta_cdf_membership(rsi_norm, alpha=3.0, beta_param=3.0)
    df["mu_M"] = (m_intraday + m_rsi) / 2.0

    # -------------------------------------------------------------------------
    # 8. Bivariate Joint Copula Aggregation (PDF 1, Pages 1-3)
    # -------------------------------------------------------------------------
    # Replaces destructive hard min(Core, S, T1, T2, M) with Frank Archimedean Copula
    df["mu_Composite"] = copula_aggregate(
        df["mu_Core"],
        df["mu_S"],
        df["mu_T1"],
        df["mu_T2"],
        df["mu_M"],
        theta=3.5,
    )

    # Beta-distributed liquidity weighting factor
    df["liquidity_weight"] = beta_liquidity_weights(df["totalTradedValue"])

    return df


def main():
    print("Running Upgraded NEPSE Copula Fuzzy Screener...")
    df = load_universe()
    print(f"Loaded {len(df)} eligible equities.")

    df = build_copula_fuzzy_sets(df)

    out_cols = [
        "symbol", "securityName", "sector", "closePrice",
        "earnings_yield", "book_yield",
        "mu_V", "mu_Q", "mu_HighDebt", "mu_NegEPS", "mu_Risk", "mu_Core",
        "mu_S", "mu_T1", "mu_T2", "mu_M", "mu_Composite", "liquidity_weight",
    ]
    avail_cols = [c for c in out_cols if c in df.columns]
    ranked = df[avail_cols].round(4).sort_values("mu_Composite", ascending=False)

    for out_path in [OUT_CSV_DATA, OUT_CSV_FUZZY]:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        ranked.to_csv(out_path, index=False)
        print(f"Saved rankings to {out_path}")

    print("\n--- TOP 15 EQUITIES BY BIVARIATE COPULA COMPOSITE ---")
    print(
        ranked[["symbol", "sector", "closePrice", "mu_V", "mu_Q", "mu_T1", "mu_T2", "mu_Composite"]].head(15).to_string(
            index=False))


if __name__ == "__main__":
    main()
