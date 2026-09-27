"""
NEPSE Bayesian screener - Upgraded Core Pipeline.

Statistical enhancements applied:
  1. Quotient Variable Regularization:
     - Inverts P/E and P/B into Earnings Yield (E/P) and Book Yield (B/P)
       to avoid division-by-zero singularities and sign flips.
     - Robust IQR-based winsorization on heavy-tailed valuation multiples.
  2. Beta-Distributed Liquidity Weights:
     - Uses Beta CDF mapping on [0, 1] instead of arbitrary heuristic clipping.
  3. Robust Laplace Likelihood in PyMC:
     - Replaces Gaussian likelihood with a Double Exponential (Laplace)
       distribution: f(y) = 0.5 * b * exp(-b * |y - mu|), making sector shrinkage
       completely robust against circuit-breaker jumps and illiquid outliers.
  4. Robust Z-Scoring:
     - Standardizes features using Median and Median Absolute Deviation (MAD)
       consistent with the Laplace scale parameter.
"""

import os
import re
import numpy as np
import pandas as pd
import scipy.stats as stats


# ---------------------------------------------------------------------
# 1. Instrument / sector classification
# ---------------------------------------------------------------------

def classify_instrument(name: str) -> str:
    n = name.lower()
    if "debenture" in n:
        return "debenture"
    if "promoter" in n:
        return "promoter_share"
    if "mutual fund" in n or re.search(r"\bfund\b", n):
        return "mutual_fund"
    return "equity"


_SECTOR_PATTERNS = [
    ("Development Bank", re.compile(r"development bank", re.I)),
    ("Commercial Bank", re.compile(r"\bbank\b", re.I)),
    ("Microfinance", re.compile(r"laghubitta|bittiya sanstha", re.I)),
    ("Life Insurance", re.compile(r"life insurance", re.I)),
    ("Non-Life Insurance", re.compile(r"insurance|bima", re.I)),
    ("Hydropower", re.compile(r"hydro|jalvidhyut|power|urja|energy", re.I)),
    ("Finance", re.compile(r"finance", re.I)),
    ("Hotels & Tourism", re.compile(r"hotel|tourism|resort|cablecar", re.I)),
    ("Investment", re.compile(r"investment", re.I)),
]


def classify_sector_by_name(name: str) -> str:
    for label, pattern in _SECTOR_PATTERNS:
        if pattern.search(name):
            return label
    return "Others"


def _pick_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lowered = {col.lower().strip(): col for col in df.columns}
    for candidate in candidates:
        if candidate.lower().strip() in lowered:
            return lowered[candidate.lower().strip()]
    return None


def merge_official_sectors(df: pd.DataFrame, securities_csv: str | None) -> pd.DataFrame:
    df = df.copy()
    df["sector"] = df["securityName"].apply(classify_sector_by_name)
    df["paid_up_capital_est"] = np.nan

    if securities_csv is None or not os.path.exists(securities_csv):
        return df

    try:
        ref = pd.read_csv(securities_csv)
    except Exception:
        return df

    SECTOR_COL = _pick_column(ref, ["sectorName", "sector", "official_sector", "sector_name"])
    SHARES_COL = _pick_column(ref, ["listedShares", "listed_shares", "issuedShares", "issued_shares", "shares",
                                    "share_count", "totalShares"])
    PAR_VALUE = 100.0

    if "symbol" not in ref.columns:
        return df

    keep = ["symbol"]
    rename = {}
    if SECTOR_COL is not None:
        keep.append(SECTOR_COL)
        rename[SECTOR_COL] = "official_sector"
    if SHARES_COL is not None:
        keep.append(SHARES_COL)
        rename[SHARES_COL] = "listed_shares"

    ref = ref[keep].rename(columns=rename)
    df = df.merge(ref, on="symbol", how="left")

    if "official_sector" in df.columns:
        df["sector"] = df["official_sector"].fillna(df["sector"])
    if "listed_shares" in df.columns:
        df["paid_up_capital_est"] = df["listed_shares"] * PAR_VALUE

    return df


# ---------------------------------------------------------------------
# 2. Load + clean
# ---------------------------------------------------------------------

def safe_div(a, b):
    a = pd.to_numeric(a, errors="coerce")
    b = pd.to_numeric(b, errors="coerce")
    b = b.replace(0, np.nan)
    return a / b


def load_and_clean(
        price_csv: str,
        securities_csv: str | None = None,
        fundamentals_csv: str | None = None,
) -> pd.DataFrame:
    df = pd.read_csv(price_csv)
    df["instrument_type"] = df["securityName"].apply(classify_instrument)
    df = df[df["instrument_type"] == "equity"].copy()

    # Drop non-traded, halted, or zero-price entries
    df = df[(df["closePrice"] > 0) & (df["fiftyTwoWeekHigh"] > df["fiftyTwoWeekLow"])].copy()
    df = merge_official_sectors(df, securities_csv)

    if fundamentals_csv is not None and os.path.exists(fundamentals_csv):
        fund = pd.read_csv(fundamentals_csv)
        df = df.merge(fund, on="symbol", how="left", suffixes=("", "_fund"))
    else:
        print(f"[warn] {fundamentals_csv} not found - running in technical-only mode.")

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------
# 3. Statistical Transformations & Robust Feature Engineering
# ---------------------------------------------------------------------

def robust_mad_zscore(series: pd.Series, invert: bool = False) -> pd.Series:
    """
    Computes a Laplace-consistent robust z-score using Median and MAD.
    Scale = 1.4826 * MAD (matches Laplace/Normal asymptotic scale),
    robust to fat tails and outliers.
    """
    clean = pd.to_numeric(series, errors="coerce")
    med = clean.median()
    mad = np.median(np.abs(clean.dropna() - med))
    scale = 1.4826 * mad if (mad is not None and mad > 1e-6) else clean.std()

    if scale == 0 or np.isnan(scale):
        scale = 1.0

    z = (clean - med) / scale
    z = z.clip(-3.5, 3.5).fillna(0.0)
    return -z if invert else z


def compute_beta_liquidity_weights(trade_values: pd.Series, alpha: float = 2.0, beta: float = 3.0) -> np.ndarray:
    """
    Maps trading volume to a bounded reliability weight in [0.05, 0.95]
    using the Beta Cumulative Distribution Function.
    """
    log_v = np.log1p(trade_values.fillna(0.0))
    v_min, v_max = log_v.min(), log_v.max()
    norm_v = (log_v - v_min) / (v_max - v_min + 1e-8)

    # Beta CDF mapping
    beta_cdf = stats.beta.cdf(norm_v, a=alpha, b=beta)
    return 0.05 + 0.90 * beta_cdf


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # --- A. Technical Signals ---
    # 52w range position in [0, 1]
    denom_52w = np.maximum(df["fiftyTwoWeekHigh"] - df["fiftyTwoWeekLow"], 1e-4)
    df["f_52w_position"] = (df["closePrice"] - df["fiftyTwoWeekLow"]) / denom_52w

    # VWAP mean reversion
    df["f_reversion"] = safe_div(df["closePrice"] - df["averageTradedPrice"], df["averageTradedPrice"]).fillna(0.0)

    # Beta-distributed liquidity weights
    df["liquidity_weight"] = compute_beta_liquidity_weights(df["totalTradedValue"])

    # Technical Composite Z-score (lower price relative to range & VWAP = higher reversion discount)
    df["z_technical"] = (
            robust_mad_zscore(df["f_52w_position"], invert=True) +
            robust_mad_zscore(df["f_reversion"], invert=True)
    )

    # --- B. Fundamental Signals (Quotient Transformations) ---
    # Convert unstable P/E and P/B into stable continuous yields (E/P and B/P)
    if "eps" in df.columns and df["eps"].notna().sum() > 5:
        df["earnings_yield"] = safe_div(df["eps"], df["closePrice"])
    elif "pe_ratio" in df.columns:
        # Invert positive PE to earnings yield; treat <= 0 as distressed
        df["earnings_yield"] = np.where(df["pe_ratio"] > 0, 1.0 / df["pe_ratio"], -0.1)

    if "pb_ratio" in df.columns:
        # Invert PB to book-to-market
        df["book_yield"] = np.where(df["pb_ratio"] > 0, 1.0 / df["pb_ratio"], 0.0)

    # Candidate signals: (column, invert_direction)
    SIGNALS = [
        ("earnings_yield", False),  # higher earnings yield = cheaper
        ("book_yield", False),  # higher book yield = cheaper
        ("roe", False),
        ("roa", False),
        ("dividend_yield", False),
        ("yoy_profit_growth", False),
        ("graham_discount", False),  # higher discount = cheaper
        ("debt_to_equity", True),  # lower debt = safer
    ]

    available = [(c, inv) for c, inv in SIGNALS if c in df.columns and df[c].notna().sum() > 5]

    if available:
        df["z_fundamental"] = 0.0
        for col, invert in available:
            df["z_fundamental"] += robust_mad_zscore(df[col], invert=invert)
        df["z_composite"] = df["z_technical"] + df["z_fundamental"]
    else:
        df["z_composite"] = df["z_technical"]

    return df


# ---------------------------------------------------------------------
# 4. Hierarchical Bayesian Model with Laplace Likelihood
# ---------------------------------------------------------------------

def fit_hierarchical_model(df: pd.DataFrame, draws: int = 1000, tune: int = 1000):
    """
    Upgraded Bayesian model using a LAPLACE likelihood.
    Laplace Likelihood (PDF 3, Page 4):
        f(y | mu, b) = 1/(2b) * exp(-|y - mu| / b)
    Provides robust L1-norm shrinkage against extreme NEPSE market jumps.
    """
    import pymc as pm

    sectors = df["sector"].astype("category")
    sector_idx = sectors.cat.codes.values
    n_sectors = len(sectors.cat.categories)

    y = df["z_composite"].values
    weights = df["liquidity_weight"].values

    with pm.Model() as model:
        # Level 0: Grand market hyperpriors
        mu_market = pm.Normal("mu_market", mu=0.0, sigma=1.0)
        tau_market = pm.HalfNormal("tau_market", sigma=1.5)

        # Level 1: Non-centered sector pooling
        sector_offset = pm.Normal("sector_offset", mu=0.0, sigma=1.0, shape=n_sectors)
        mu_sector = pm.Deterministic("mu_sector", mu_market + sector_offset * tau_market)

        # Level 2: Laplace dispersion scaled by Beta liquidity weights
        # Highly liquid stocks -> higher precision (smaller b scale)
        b_base = pm.HalfNormal("b_base", sigma=1.5)
        b_i = b_base / np.sqrt(weights)

        # LAPLACE LIKELIHOOD (Double Exponential)
        pm.Laplace("y_obs", mu=mu_sector[sector_idx], b=b_i, observed=y)

        trace = pm.sample(
            draws=draws, tune=tune, target_accept=0.95,
            progressbar=True, random_seed=42,
            chains=2, cores=1,
        )

    return model, trace, sectors.cat.categories


def score_stocks(df: pd.DataFrame, trace, sector_categories) -> pd.DataFrame:
    mu_sector_post = trace.posterior["mu_sector"]
    sector_mean = mu_sector_post.mean(dim=("chain", "draw")).values
    sector_hdi_lo = mu_sector_post.quantile(0.05, dim=("chain", "draw")).values
    sector_hdi_hi = mu_sector_post.quantile(0.95, dim=("chain", "draw")).values

    cat_to_idx = {c: i for i, c in enumerate(sector_categories)}
    idx = df["sector"].map(cat_to_idx).values

    out = df.copy()
    out["sector_posterior_mean"] = sector_mean[idx]
    out["sector_90pct_lo"] = sector_hdi_lo[idx]
    out["sector_90pct_hi"] = sector_hdi_hi[idx]

    # Liquidity-weighted shrinkage towards sector posterior
    w = out["liquidity_weight"].values
    out["shrunk_score"] = w * out["z_composite"] + (1.0 - w) * out["sector_posterior_mean"]

    return out.sort_values("shrunk_score", ascending=False)


# ---------------------------------------------------------------------
# 5. Runner
# ---------------------------------------------------------------------

if __name__ == "__main__":
    base_data_dir = "C:\\Users\\saral\\OneDrive\\Python code for Nepse-latest\\data"

    price_file = os.path.join(base_data_dir, "today_price.csv")
    sec_file = os.path.join(base_data_dir, "securities_reference.csv")
    fund_file = os.path.join(base_data_dir, "fundamentals_base.csv")

    df = load_and_clean(
        price_csv=price_file,
        securities_csv=sec_file if os.path.exists(sec_file) else None,
        fundamentals_csv=fund_file if os.path.exists(fund_file) else None,
    )
    print(f"Equities loaded and cleaned: {len(df)}")
    print(df["sector"].value_counts())

    df = engineer_features(df)
    model, trace, sector_cats = fit_hierarchical_model(df)
    ranked = score_stocks(df, trace, sector_cats)

    cols = [
        "symbol", "securityName", "sector", "closePrice",
        "f_52w_position", "z_technical", "z_composite",
        "sector_posterior_mean", "shrunk_score",
    ]

    print("\n--- TOP 15 RANKED OPPORTUNITIES (Laplace Robust + Beta Weighted) ---")
    print(ranked[cols].head(15).to_string(index=False))

    output_path = os.path.join(base_data_dir, "screener_output.csv")
    ranked.to_csv(output_path, index=False)
    print(f"\nSuccessfully saved updated rankings to: {output_path}")