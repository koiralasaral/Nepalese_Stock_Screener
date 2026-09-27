import pandas as pd
import numpy as np
import warnings
from nepse_scraper import NepseScraper

warnings.filterwarnings('ignore')


def generate_volatility_file(csv_path=r"C:\Users\saral\PycharmProjects\PythonProject1\data\today_price.csv",
                             output_path=r"/data/realized_volatility.csv", days=1000):
    print(f"Loading stock list from {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: Could not find {csv_path}. Make sure you are running this in the correct folder.")
        return

    if 'securityId' not in df.columns:
        print("Error: 'securityId' column is missing from today_price.csv. Cannot fetch historical data.")
        return

    # Filter out empty security IDs and get list of tuples
    valid_stocks = df[df['securityId'].notna()]
    target_assets = list(zip(valid_stocks['symbol'], valid_stocks['securityId']))

    print(f"Attempting to fetch historical data for {len(target_assets)} stocks...")
    scraper = NepseScraper(verify_ssl=False)

    def extract_records(payload):
        if isinstance(payload, list): return payload
        if isinstance(payload, dict):
            for key in ("content", "data", "payload", "result"):
                records = extract_records(payload.get(key))
                if records: return records
            if any(key in payload for key in ("closePrice", "closingPrice", "lastTradedPrice", "lastUpdatedPrice")):
                return [payload]
        return []

    returns_dict = {}

    for sym, security_id in target_assets:
        # Cast security_id to int if it's stored as a float
        sec_id = int(security_id)
        print(f"  Fetching price history for {sym} (securityId={sec_id})...")

        try:
            endpoint_name = f"security_price_{sec_id}"
            scraper.register_endpoint(
                name=endpoint_name,
                path=f"/api/nots/market/security/price/{sec_id}",
                method="GET"
            )
            response = scraper.call_endpoint(endpoint_name, params={"page": 0, "size": days + 1})
            records = extract_records(response)

            rows = []
            for rec in records:
                if not isinstance(rec, dict): continue
                price = None
                for key in ("closePrice", "closingPrice", "lastTradedPrice", "lastUpdatedPrice", "ltp"):
                    if rec.get(key) not in (None, ""):
                        price = pd.to_numeric(rec.get(key), errors="coerce")
                        break

                # Check for valid price and business date
                if price is not None and np.isfinite(price) and price > 0:
                    trade_date = (rec.get("businessDate") or rec.get("date") or rec.get("tradeDate") or rec.get(
                        "createdDate"))
                    rows.append({"date": trade_date, "price": float(price)})

            if len(rows) < 5:
                print(f"    Skipping {sym}: Not enough valid data points.")
                continue

            # Convert to DataFrame, sort by date to ensure chronological order
            price_df = pd.DataFrame(rows)
            price_df["date"] = pd.to_datetime(price_df["date"], errors="coerce")
            price_df = price_df.sort_values("date")

            # Calculate daily returns
            prices = price_df["price"]
            returns = prices.pct_change().replace([np.inf, -np.inf], np.nan).dropna()

            if len(returns) >= 5:
                returns_dict[sym] = returns.tail(days).reset_index(drop=True)

        except Exception as e:
            print(f"    Failed {sym}: {e}")

    if not returns_dict:
        print("\nCritical Error: No historical price data could be retrieved for any stock.")
        return

    # Build Dataframe of returns (filling unequal lengths with NaN)
    returns_df = pd.DataFrame(returns_dict)

    # Calculate annualized standard deviation (volatility)
    # Assumes ~252 trading days in a year
    volatility_series = returns_df.std(skipna=True) * np.sqrt(252)

    # Format into DataFrame for merging
    vol_df = volatility_series.reset_index()
    vol_df.columns = ["symbol", "realized_volatility"]

    # Save to CSV
    vol_df.to_csv(output_path, index=False)
    print(f"\nSuccessfully generated '{output_path}' with real volatility for {len(vol_df)} stocks!")


if __name__ == "__main__":
    # Ensure this script is looking in the correct local directory
    generate_volatility_file(csv_path=r"C:\Users\saral\PycharmProjects\PythonProject1\data\today_price.csv",
                             output_path="data/realized_volatility.csv", days=180)