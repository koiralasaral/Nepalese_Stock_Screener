"""
Back-derive stable per-share fundamentals (EPS, book value, dividend/share,
implied EPS growth rate) from the ratio snapshot already paid for in
fundamentals.csv, using the price that was current at the time that
snapshot was pulled.

Why this is safe to do locally, no Parse.bot credits required:
  pe_ratio  = price / eps                    -> eps          = price / pe_ratio
  pb_ratio  = price / book_value_per_share    -> book_value   = price / pb_ratio
  dividend_yield = dividend_per_share/price*100 -> dividend_per_share = dividend_yield/100 * price
  peg_ratio = pe_ratio / eps_growth_pct        -> eps_growth_pct = pe_ratio / peg_ratio
  graham_number = sqrt(22.5 * eps * book_value)              (stable once eps/book_value are stable)

roe, roa, payout_ratio are accounting ratios (not price-dependent) and are
carried over unchanged.

Output: fundamentals_base.csv - the "slow" layer, re-derive only when you
refetch real fundamentals (new quarter). Ratios (pe/pb/dividend_yield/peg/
graham_discount) are the "fast" layer, recomputed every trading day from
this base + today_price.csv, with update_ratios.py - no Parse.bot call.
"""
import numpy as np
import pandas as pd

FUND_CSV = r"C:\Users\saral\PycharmProjects\PythonProject1\data\fundamentals.csv"
PRICE_CSV = r"C:\Users\saral\PycharmProjects\PythonProject1\data\today_price.csv"
OUT_CSV = r"C:\Users\saral\PycharmProjects\PythonProject1\data\fundamentals_base.csv"


def safe_div(a, b):
    b = b.replace(0, np.nan)
    return a / b


def build():
    fund = pd.read_csv(FUND_CSV)
    price = pd.read_csv(PRICE_CSV)[["symbol", "closePrice", "businessDate"]]

    df = fund.merge(price, on="symbol", how="left")
    missing_price = df["closePrice"].isna().sum()
    if missing_price:
        print(f"[warn] {missing_price} symbols in fundamentals.csv had no "
              f"matching price row in today_price.csv - eps/book_value will "
              f"be NaN for those until a price is available.")

    df["eps"] = safe_div(df["closePrice"], df["pe_ratio"])
    df["book_value"] = safe_div(df["closePrice"], df["pb_ratio"])
    df["dividend_per_share"] = df["dividend_yield"] / 100.0 * df["closePrice"]
    df["eps_growth_pct"] = safe_div(df["pe_ratio"], df["peg_ratio"])

    graham_raw = 22.5 * df["eps"] * df["book_value"]
    df["graham_number"] = np.sqrt(graham_raw.clip(lower=0))
    df.loc[graham_raw < 0, "graham_number"] = np.nan

    # sanity check against the graham_discount Parse.bot already gave us
    check = (df["closePrice"] - df["graham_number"]) / df["graham_number"] * 100
    df["_graham_discount_check"] = check.round(0)
    mismatch = (
        (df["graham_discount"].notna())
        & (df["_graham_discount_check"].notna())
        & ((df["_graham_discount_check"] - df["graham_discount"]).abs() > 5)
    )
    print(f"Rows with graham_discount mismatch >5 vs re-derivation: {mismatch.sum()} / {df['graham_discount'].notna().sum()}")

    base_cols = [
        "symbol", "eps", "book_value", "dividend_per_share", "eps_growth_pct",
        "graham_number", "roe", "roa", "payout_ratio",
    ]
    if "debt_to_equity" in df.columns:
        base_cols.append("debt_to_equity")
    audit_cols = ["closePrice", "pe_ratio", "pb_ratio", "peg_ratio",
                  "dividend_yield", "graham_discount", "businessDate"]

    base = df[base_cols + audit_cols].rename(columns={
        "closePrice": "fetch_price",
        "pe_ratio": "fetch_pe_ratio",
        "pb_ratio": "fetch_pb_ratio",
        "peg_ratio": "fetch_peg_ratio",
        "dividend_yield": "fetch_dividend_yield",
        "graham_discount": "fetch_graham_discount",
        "businessDate": "fetch_date",
    })

    import os
    os.makedirs(r"C:\Users\saral\PycharmProjects\PythonProject1\data", exist_ok=True)
    base.to_csv(OUT_CSV, index=False)
    print(f"\nWrote {OUT_CSV} - {len(base)} rows")
    print(base[["symbol", "eps", "book_value", "dividend_per_share", "eps_growth_pct", "graham_number"]].head(10).to_string(index=False))
    return base


if __name__ == "__main__":
    build()
