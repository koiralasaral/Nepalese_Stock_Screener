"""
Data acquisition layer - run this locally where nepse_scraper can reach
nepalstock.com (this sandbox's network is allowlisted to package registries
only, so none of this can be executed or verified here).

Pulls THREE things nepse_scraper can actually provide, honestly:

  1. today_price.csv           - price/OHLCV/mcap/52w range (you already had this)
  2. securities_reference.csv  - official company master list: symbol,
                                  sector tag, listed/issued shares
                                  (get_all_securities + get_sectors)
  3. disclosures.csv           - links to filed quarterly/annual reports
                                  (get_company_disclosures) - these are the
                                  PDFs that actually contain paid-up capital,
                                  book value, debt-to-equity, and YoY profit
                                  growth. nepse_scraper gives you the filing
                                  index, not parsed numbers - a PDF parser is
                                  a separate piece of work.

It deliberately does NOT attempt to produce PE/PB/ROE/ROA/PEG/dividend
yield/payout ratio/Graham number - nepse_scraper has no endpoint for these
because nepalstock.com's own API doesn't compute them. See
fetch_fundamentals_stub() at the bottom for where a real external source
(NepseAlpha, MeroLagani, or your own PDF-parsed disclosures) plugs in.
"""

import os
import warnings
import pandas as pd
from nepse_scraper import NepseScraper

warnings.filterwarnings("ignore", category=Warning)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE_DIR, "data")


def fetch_reference_data(scraper: NepseScraper):
    """Official sector tags + listed share counts. Real nepse_scraper calls."""
    securities = scraper.get_all_securities()
    sectors = scraper.get_sectors()

    sec_df = pd.DataFrame(securities)
    sector_df = pd.DataFrame(sectors)

    print("get_all_securities() columns:", sec_df.columns.tolist())
    print("get_sectors() columns:", sector_df.columns.tolist())
    print(
        "\n-> build_screener.py's merge_official_sectors() guesses "
        "'sectorName' and 'listedShares' as the column names. Compare "
        "against the printout above and fix those two constants if the "
        "real API uses different names."
    )

    os.makedirs(OUT, exist_ok=True)
    sec_df.to_csv(os.path.join(OUT, "securities_reference.csv"), index=False)
    return sec_df, sector_df


def fetch_disclosures(scraper: NepseScraper):
    """Filing index only - not parsed financials. This is the honest path
    to paid-up capital / book value / D-E / YoY growth: the numbers live
    inside these filed PDFs, nepalstock.com doesn't expose them as fields."""
    disclosures = scraper.get_company_disclosures()
    disc_df = pd.DataFrame(disclosures)
    os.makedirs(OUT, exist_ok=True)
    disc_df.to_csv(os.path.join(OUT, "disclosures.csv"), index=False)
    print(f"disclosures: {len(disc_df)} rows (filing links, not parsed numbers)")
    return disc_df


def fetch_fundamentals_stub():
    """Populate fundamentals.csv via the Parse API-backed downloader."""
    from fetch_fundamentals import build_fundamentals_csv

    price_path = os.path.join(OUT, "today_price.csv")
    if not os.path.exists(price_path):
        raise FileNotFoundError(f"{price_path} not found; run fetch_all() first")

    price_df = pd.read_csv(price_path)
    symbols = [str(sym).strip() for sym in price_df.get("symbol", []).tolist() if str(sym).strip()]
    build_fundamentals_csv(symbols)


def fetch_all():
    os.makedirs(OUT, exist_ok=True)
    scraper = NepseScraper(verify_ssl=False)

    prices = scraper.get_today_price()
    price_df = pd.DataFrame(prices)
    os.makedirs(OUT, exist_ok=True)
    price_df.to_csv(os.path.join(OUT, "today_price.csv"), index=False)
    print(f"today_price: {len(price_df)} rows")

    fetch_reference_data(scraper)
    #fetch_disclosures(scraper)
    #fetch_fundamentals_stub()

    print(f"\nDone. Wrote outputs to {OUT}")


if __name__ == "__main__":
    fetch_all()
