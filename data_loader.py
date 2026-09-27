"""
data_loader.py
===============
Loads and merges the NEPSE screener CSV exports into one clean
DataFrame ready for Heston Monte Carlo, with the Heston inputs
(S0, v0, theta, mu) already attached.

Base table: screener_output.csv -- it's already filtered to
instrument_type == 'equity' and already carries your existing
Bayesian-shrunk composite (`shrunk_score`) and fundamental/technical
z-scores, so it's the natural spine to merge onto.

Filtering: rows with eps == NaN are dropped. In this data that's
mutual-fund / closed-end-fund units trading on the equity board near
their NPR 10 par value (no EPS, no PE) -- not operating companies, so
they don't belong in a stock-specific Heston screener. Everything with
a real EPS is kept.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path


REQUIRED_FILES = ["screener_output.csv", "realized_volatility.csv"]


def load_universe(data_dir: str | Path, mu_clip: tuple[float, float] = (-0.30, 0.60)) -> pd.DataFrame:
    data_dir = Path(data_dir)
    for f in REQUIRED_FILES:
        if not (data_dir / f).exists():
            raise FileNotFoundError(f"Missing required file: {data_dir / f}")

    so = pd.read_csv(data_dir / "screener_output.csv")
    rv = pd.read_csv(data_dir / "realized_volatility.csv")

    df = so.merge(rv, on="symbol", how="left", validate="m:1")

    # Keep real operating companies only (drop fund units etc. with no EPS)
    df = df[df["eps"].notna()].copy()

    # Need a usable current price and a usable (real) volatility
    df["S0"] = df["fetch_price"].fillna(df["closePrice"])
    df = df[df["S0"].notna() & (df["S0"] > 0)]
    df = df[df["realized_volatility"].notna() & (df["realized_volatility"] > 0)]

    # --- Heston inputs -----------------------------------------------
    # v0 = theta = realized_volatility^2 : the only assumption a single
    # trailing realized-vol scalar supports (see heston_model.py docstring).
    df["theta"] = df["realized_volatility"] ** 2
    df["v0"] = df["theta"]

    # Expected-return input (mu): Graham-style earnings yield as the
    # forward-return proxy, since it's already computed in your
    # pipeline (screener_output.earnings_yield = eps/price-derived).
    # Falls back to 1/PE, then to the cross-sectional median for names
    # where PE is negative/undefined (loss-making or distressed), and
    # is clipped to a sane band so a handful of extreme PE/EPS outliers
    # can't dominate the simulated drift.
    ey = df["earnings_yield"].copy()
    fallback = 1.0 / df["fetch_pe_ratio"].replace(0, np.nan)
    ey = ey.where(ey.notna() & (ey != 0), fallback)
    positive_ey = ey[ey > 0]
    median_ey = positive_ey.median() if len(positive_ey) else 0.08
    df["mu"] = ey.where(ey.notna() & (ey > 0), median_ey)
    df["mu"] = df["mu"].clip(*mu_clip)

    keep_cols = [
        "symbol", "securityName", "sector", "S0", "closePrice",
        "realized_volatility", "v0", "theta", "mu",
        "earnings_yield", "fetch_pe_ratio", "fetch_pb_ratio", "roe", "roa",
        "debt_to_equity", "eps", "eps_growth_pct", "dividend_per_share",
        "graham_number", "f_52w_position", "f_reversion", "z_technical",
        "z_fundamental", "z_composite", "shrunk_score", "liquidity_weight",
        "sector_posterior_mean",
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols].reset_index(drop=True)
    return df


def load_market_index_series(data_dir: str | Path, index_name: str = "NEPSE") -> pd.Series | None:
    """
    Pull the short daily index sparkline embedded in sector_ratios.csv
    (currently ~16 trading days for the NEPSE composite). Useful as an
    optional, clearly-low-confidence input to
    heston_model.calibrate_heston_mom() for a market-wide dynamics
    cross-check -- NOT a substitute for per-symbol history.
    """
    import ast
    data_dir = Path(data_dir)
    path = data_dir / "sector_ratios.csv"
    if not path.exists():
        return None
    sec = pd.read_csv(path)
    row = sec[sec["index_name"] == index_name]
    if row.empty or "sparkline" not in sec.columns:
        return None
    spark = ast.literal_eval(row.iloc[0]["sparkline"])
    return pd.Series(spark["current"], index=pd.to_datetime(spark["time"]))
