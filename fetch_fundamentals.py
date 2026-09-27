"""
Fundamentals acquisition - fills the gap nepse_scraper can't.

Three sources, three different reliability levels - read the notes on
each before trusting its output blindly, since NONE of this can be
executed or verified from this sandbox (network here only reaches package
registries, not api.parse.bot or merolagani.com). Run it locally, then
read the printed column/structure diagnostics before using the output.

  A. NepseAlpha via Parse.bot  -> PE, PB, ROE, ROA, PEG, dividend yield,
     payout ratio, Graham number discount, all stocks in ONE call each
     (get_fundamental_signals, get_sector_summary). Needs a Parse API key:
     sign up free at https://parse.bot (100 credits/month), then
       export PARSE_API_KEY=...
     RELIABILITY: high for the numbers, but the exact JSON field names
     below are my best reading of Parse's prose documentation, not a
     confirmed schema sample - the code prints the raw response once so
     you can fix FUND_FIELD_MAP / SECTOR_FIELD_MAP in one place if the
     real keys differ.

  B. MeroLagani via Parse.bot  -> EPS per symbol via get_company_details.
     This is ONE credit PER SYMBOL (documented pricing), so 100 symbols =
     100 credits = your whole free-tier month. Sleeps between calls to
     respect the 5 req/min free-tier rate limit - budget ~20 minutes for
     100 symbols on the free tier, or upgrade to Hobby ($30/mo, 20/min)
     if you want this to run in ~5 minutes.
     RELIABILITY: medium - MeroLagani's documented response for this
     endpoint does NOT list a book-value field (only ltp, eps, peRatio,
     marketCap, fiftyTwoHigh, fiftyTwoLow, percentChange), so book value
     is NOT pulled from here - see source C instead.

  C. Direct HTML scrape of merolagani.com's company detail page ->
     paid-up capital, book value, debt-to-equity, YoY profit growth.
     These are NOT in any of the above API responses (neither Parse.bot
     wrapper documents them as fields) - the full merolagani.com webpage
     shows them in a "Key Indicators" / "Capital Structure" panel that
     the API endpoints don't expose. This is a real, unauthenticated
     requests.get() + BeautifulSoup scrape, no API key needed, but it IS
     fragile: label text matching, not a stable JSON contract, and the
     exact label wording is a best guess until you run it once and check
     the printed diagnostics.
"""

import os
import re
import time
import requests
import pandas as pd
from bs4 import BeautifulSoup

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
PARSE_API_BASE = "https://api.parse.bot/scraper"
NEPSEALPHA_UUID = "c91857fd-7333-4fa5-bbc1-bab762e4137b"
MEROLAGANI_UUID = "e0e96569-8b5b-4bb5-9680-0d5927228350"


def _parse_headers():
    key = "pmx_de1408d7f300cc6a01b3eb352266f6d4"
    if not key:
        raise RuntimeError(
            "Set PARSE_API_KEY first (sign up free at https://parse.bot). "
            "export PARSE_API_KEY=... then re-run."
        )
    return {"X-API-Key": key}


# ---------------------------------------------------------------------
# A. NepseAlpha bulk ratios + sector summary
# ---------------------------------------------------------------------

FUND_FIELD_MAP = {
    "pe_ratio": "PE",
    "pb_ratio": "PB",
    "roe": "ROE",
    "roa": "ROA",
    "peg_ratio": "PEG",
    "dividend_yield": "Dividend Yield",
    "payout_ratio": "Payout Ratio",
    "graham_discount": "Discount From Graham Number",
    "debt_to_equity": "Debt To Equity",
    "borrowings_to_equity": "Borrowings To Equity",
    "total_liabilities": "Total Liabilities",
    "borrowings": "Borrowings",
    "current_liabilities": "Current Liabilities",
    "total_equity": "Total Equity",
    "total_assets": "Total Assets",
}

SECTOR_FIELD_MAP = {
    "sector_pe": "pe",
    "sector_pb": "pb",
    "sector_roe": "roe",
}


def _pick_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lowered = {col.lower().strip(): col for col in df.columns}
    for candidate in candidates:
        if candidate.lower().strip() in lowered:
            return lowered[candidate.lower().strip()]
    return None


def _coerce_records(payload):
    if isinstance(payload, dict):
        for key in ("data", "results"):
            if key in payload and isinstance(payload[key], (dict, list)):
                return _coerce_records(payload[key])
        for key in ("signals", "home_table"):
            if key in payload and isinstance(payload[key], list):
                return payload[key]
        return [payload]
    if isinstance(payload, list):
        return payload
    return [payload]


def _to_numeric(value):
    if pd.isna(value):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip().replace(",", "").replace("%", "")
        if not text or text in {"-", "--", "na", "n/a"}:
            return None
        try:
            return float(text)
        except ValueError:
            return None
    return None


def fetch_nepsealpha_signals() -> pd.DataFrame:
    headers = _parse_headers()
    resp = requests.get(
        f"{PARSE_API_BASE}/{NEPSEALPHA_UUID}/get_fundamental_signals",
        headers=headers, params={"include_debt": True}, timeout=120,
    )
    resp.raise_for_status()
    payload = resp.json()
    print("get_fundamental_signals raw top-level keys:", list(payload.keys()))

    records = _coerce_records(payload)
    df = pd.DataFrame(records)
    if df.empty:
        print("[warn] get_fundamental_signals returned no rows")
        return pd.DataFrame(columns=["symbol"])

    print("get_fundamental_signals columns:", df.columns.tolist())

    rename = {}
    for dest, source in FUND_FIELD_MAP.items():
        src_col = _pick_column(df, [source, source.lower(), source.title()])
        if src_col is not None:
            rename[src_col] = dest

    for old_col, new_col in rename.items():
        df[new_col] = df[old_col].apply(_to_numeric)

    symbol_col = _pick_column(df, ["symbol", "Symbol"])
    if symbol_col is None:
        symbol_col = df.columns[0]
    df = df.rename(columns={symbol_col: "symbol"})
    if "symbol" in df.columns:
        df["symbol"] = df["symbol"].astype(str).str.strip()

    for col in list(rename.values()):
        if col != "symbol" and col not in df.columns:
            df[col] = None

    return df[["symbol"] + [c for c in rename.values() if c != "symbol"]]


def fetch_nepsealpha_sector_summary() -> pd.DataFrame:
    headers = _parse_headers()
    resp = requests.get(
        f"{PARSE_API_BASE}/{NEPSEALPHA_UUID}/get_sector_summary",
        headers=headers, timeout=120,
    )
    resp.raise_for_status()
    payload = resp.json()
    print("get_sector_summary raw top-level keys:", list(payload.keys()))

    records = _coerce_records(payload)
    df = pd.DataFrame(records)
    print("get_sector_summary columns:", df.columns.tolist())

    rename = {}
    for dest, source in SECTOR_FIELD_MAP.items():
        src_col = _pick_column(df, [source, source.lower(), source.title()])
        if src_col is not None:
            rename[src_col] = dest
    if rename:
        df = df.rename(columns=rename)
    return df


# ---------------------------------------------------------------------
# B. MeroLagani per-symbol EPS loop
# ---------------------------------------------------------------------

def fetch_merolagani_eps(symbols: list[str], sleep_s: float = 12.0) -> pd.DataFrame:
    """1 credit per symbol - sleep_s=12 respects the 5 req/min free tier.
    Drop to ~3s if you're on the Hobby tier (20 req/min)."""
    headers = _parse_headers()
    rows = []
    for i, sym in enumerate(symbols):
        try:
            resp = requests.get(
                f"{PARSE_API_BASE}/{MEROLAGANI_UUID}/get_company_details",
                headers=headers, params={"symbol": sym}, timeout=30,
            )
            resp.raise_for_status()
            payload = resp.json()
            data = payload.get("data", payload)
            rows.append({
                "symbol": sym,
                "eps": data.get("eps"),
                "pe_ratio_merolagani": data.get("peRatio"),
            })
        except Exception as e:
            print(f"  [skip] {sym}: {e}")
        if i % 10 == 0:
            print(f"  merolagani eps {i}/{len(symbols)}")
        time.sleep(sleep_s)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# C. Direct HTML scrape - paid-up capital, book value, D/E, YoY growth
# ---------------------------------------------------------------------

# Label text -> output column. Matching is fuzzy (substring, case-
# insensitive) since the exact wording on the live page is unconfirmed.
LABEL_MAP = {
    "paid up capital": "paid_up_capital",
    "paid-up capital": "paid_up_capital",
    "book value": "book_value",
    "debt to equity": "debt_to_equity",
    "debt/equity": "debt_to_equity",
    "net profit growth": "yoy_profit_growth",
    "profit growth": "yoy_profit_growth",
    "return on equity": "roe_html_backup",
}

_NUMERIC_RE = re.compile(r"-?[\d,]+\.?\d*")


def _extract_number(text: str):
    m = _NUMERIC_RE.search(text.replace(",", ""))
    return float(m.group()) if m else None


def scrape_merolagani_company_page(symbol: str, sleep_s: float = 2.0) -> dict:
    """No API key needed - plain unauthenticated page fetch. Fragile:
    depends on merolagani.com's live HTML structure, which this sandbox
    cannot check. Run scrape_one_and_inspect() below FIRST against a
    single real symbol and read the printed table dump before trusting
    this on all 100."""
    url = f"https://merolagani.com/CompanyDetail.aspx?symbol={symbol}"
    resp = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    result = {"symbol": symbol}
    for row in soup.find_all(["tr", "li", "div"]):
        row_text = row.get_text(" ", strip=True).lower()
        for label, col in LABEL_MAP.items():
            if label in row_text and col not in result:
                val = _extract_number(row_text)
                if val is not None:
                    result[col] = val

    time.sleep(sleep_s)
    return result


def scrape_one_and_inspect(symbol: str = "NABIL"):
    """Run this FIRST, standalone, before looping 100 symbols. Dumps every
    table on the page so you can see the real label wording and fix
    LABEL_MAP above if matches come back empty."""
    url = f"https://merolagani.com/CompanyDetail.aspx?symbol={symbol}"
    resp = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    for i, table in enumerate(soup.find_all("table")):
        print(f"--- table {i} ---")
        print(table.get_text(" | ", strip=True)[:500])
    parsed = scrape_merolagani_company_page(symbol)
    print("\nParsed with current LABEL_MAP:", parsed)
    return parsed


def scrape_merolagani_all(symbols: list[str]) -> pd.DataFrame:
    rows = []
    for i, sym in enumerate(symbols):
        try:
            rows.append(scrape_merolagani_company_page(sym))
        except Exception as e:
            print(f"  [skip] {sym}: {e}")
        if i % 10 == 0:
            print(f"  scraping {i}/{len(symbols)}")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------

def build_fundamentals_csv(symbols: list[str]):
    os.makedirs(OUT, exist_ok=True)

    print("=== A. NepseAlpha bulk ratios ===")
    fund = fetch_nepsealpha_signals()
    sector = fetch_nepsealpha_sector_summary()
    sector.to_csv(os.path.join(OUT, "sector_ratios.csv"), index=False)

    if "symbol" in fund.columns:
        fund = fund.drop_duplicates(subset=["symbol"]).copy()

    requested_symbols = [str(s).strip() for s in symbols if str(s).strip()]
    if requested_symbols and "symbol" in fund.columns:
        fund = fund[fund["symbol"].isin(requested_symbols)].copy()

    eps_df = pd.DataFrame(columns=["symbol", "eps"])
    html_df = pd.DataFrame(columns=["symbol"])

    merged = fund.merge(eps_df, on="symbol", how="outer") \
                 .merge(html_df, on="symbol", how="outer")
    fundamentals_path = os.path.join(OUT, "fundamentals.csv")
    try:
        merged.to_csv(fundamentals_path, index=False)
    except PermissionError:
        fundamentals_path = os.path.join(OUT, "fundamentals_with_debt.csv")
        merged.to_csv(fundamentals_path, index=False)
        print(f"[warn] fundamentals.csv is locked; saved to {fundamentals_path}")
    print(f"\nWrote {fundamentals_path} - {len(merged)} rows, "
          f"columns: {merged.columns.tolist()}")
    return merged


if __name__ == "__main__":
    # symbols come from your existing equity list - reuse it rather than
    # retyping 100 tickers
    from build_screener import classify_instrument
    price_df = pd.read_csv(r"C:\Users\saral\PycharmProjects\PythonProject1\data\today_price.csv")
    price_df["instrument_type"] = price_df["securityName"].apply(classify_instrument)
    equity_symbols = price_df.loc[
        price_df["instrument_type"] == "equity", "symbol"
    ].tolist()

    build_fundamentals_csv(equity_symbols)
