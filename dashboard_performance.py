"""
dashboard_performance.py
==========================
Builds the SECOND dashboard: actual daily closing price for the last
180 calendar days, for the top-10 symbols the screener picked.

DATA REQUIREMENT -- read this first
------------------------------------
None of the files you uploaded (fundamentals.csv, fundamentals_base.csv,
realized_volatility.csv, screener_output.csv, sector_ratios.csv,
securities_reference.csv, today_price.csv, nepse_fuzzy_screener.csv)
contain a per-symbol daily price HISTORY -- only a single latest price
per stock plus one pre-computed realized_volatility number. There's
nothing to honestly plot here yet: this script will not invent 180 days
of made-up closing prices.

What it needs: a CSV with one row per (symbol, date), long format:

    symbol,date,close
    NLO,2026-03-31,238.0
    NLO,2026-04-01,241.5
    ...

That's exactly the shape your existing NepseScraper-based historical
fetch (mentioned in your notes, used previously for realized-vol/
technical-indicator work) should be able to emit -- point its output
at PRICE_HISTORY_CSV below (or pass --history-csv), and this script
does the rest: filters to the top-10 symbols, keeps the last 180
calendar days per symbol, and renders the dashboard.

If the file isn't found, this script still writes a valid HTML file
that says so plainly (rather than failing silently or faking data),
so your pipeline doesn't break -- it just won't have a chart yet.

Run: python dashboard_performance.py --out out --history-csv data/price_history.csv
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

LOOKBACK_DAYS = 180

PLACEHOLDER_HTML = """<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<title>NEPSE Past Performance -- data needed</title>
<style>
 body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;background:#0f1420;color:#e8ecf6;
      display:flex;align-items:center;justify-content:center;min-height:100vh;margin:0;padding:24px;}}
 .card{{max-width:680px;background:#161d2e;border:1px solid #232b40;border-radius:12px;padding:28px 32px;}}
 h1{{font-size:20px;margin-top:0;}}
 code{{background:#1b2740;padding:2px 6px;border-radius:4px;}}
 table{{width:100%;border-collapse:collapse;margin-top:14px;font-size:13px;}}
 td,th{{padding:6px 8px;border-bottom:1px solid #232b40;text-align:left;}}
</style></head><body>
<div class="card">
  <h1>No 180-day price history found</h1>
  <p>This dashboard needs a daily-close time series per symbol, which wasn't
     among the files you uploaded (only a single latest price + one
     realized-volatility number per stock was available). Nothing was
     invented to fill this in.</p>
  <p>Point <code>--history-csv</code> at a file shaped like:</p>
  <table>
    <tr><th>symbol</th><th>date</th><th>close</th></tr>
    <tr><td>{sym0}</td><td>2026-03-31</td><td>238.0</td></tr>
    <tr><td>{sym0}</td><td>2026-04-01</td><td>241.5</td></tr>
    <tr><td>...</td><td>...</td><td>...</td></tr>
  </table>
  <p style="margin-top:14px;color:#93a0bf;">Top-10 symbols this dashboard is waiting for:
     <b>{symbols}</b></p>
</div>
</body></html>"""


def load_price_history(path: Path, symbols: list[str]) -> pd.DataFrame | None:
    if not path.exists():
        return None
    hist = pd.read_csv(path, parse_dates=["date"])
    missing_cols = {"symbol", "date", "close"} - set(hist.columns)
    if missing_cols:
        raise ValueError(f"{path} is missing required column(s): {missing_cols}")
    hist = hist[hist["symbol"].isin(symbols)].sort_values(["symbol", "date"])
    cutoff = hist["date"].max() - pd.Timedelta(days=LOOKBACK_DAYS)
    hist = hist[hist["date"] >= cutoff]
    return hist


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>NEPSE Top 10 -- 180-Day Past Performance</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.5.1"></script>
<style>
  :root{ --bg:#0f1420; --bg-card:#161d2e; --bg-header:#0b0f1a; --border:#232b40; --text:#e8ecf6; --text-secondary:#93a0bf; --positive:#22c58b; --negative:#ff5c7c; --radius:12px; --gap:16px; }
  *{box-sizing:border-box;}
  body{margin:0;background:var(--bg);color:var(--text);font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;}
  .wrap{max-width:1320px;margin:0 auto;padding:20px;}
  header{background:var(--bg-header);border:1px solid var(--border);border-radius:var(--radius);padding:20px 24px;margin-bottom:var(--gap);}
  header h1{font-size:20px;margin:0 0 4px 0;}
  header p{margin:0;color:var(--text-secondary);font-size:13px;}
  .kpi-row{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:var(--gap);margin-bottom:var(--gap);}
  .kpi-card{background:var(--bg-card);border:1px solid var(--border);border-radius:var(--radius);padding:16px 20px;}
  .kpi-label{font-size:12px;color:var(--text-secondary);text-transform:uppercase;letter-spacing:.5px;margin-bottom:6px;}
  .kpi-value{font-size:22px;font-weight:700;}
  .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:var(--gap);}
  .chart-container{background:var(--bg-card);border:1px solid var(--border);border-radius:var(--radius);padding:16px 18px;}
  .chart-container h3{font-size:14px;margin:0 0 10px 0;display:flex;justify-content:space-between;}
  .chart-container canvas{max-height:260px;}
  .pos{color:var(--positive);} .neg{color:var(--negative);}
  footer{text-align:center;color:var(--text-secondary);font-size:12px;padding:20px 0;}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>Top 10 -- 180-Day Price History</h1>
    <p>Daily close, actual historical data &middot; generated <span id="gen-time"></span></p>
  </header>
  <section class="kpi-row" id="kpi-row"></section>
  <section class="grid" id="chart-grid"></section>
  <footer>NEPSE Top 10 Past Performance &middot; actual daily closes, not simulated</footer>
</div>
<script>
const DATA = __DATA_JSON__;
document.getElementById('gen-time').textContent = DATA.meta.generated_at;

const nGainers = DATA.series.filter(s => s.period_return >= 0).length;
document.getElementById('kpi-row').innerHTML = [
  ['Symbols shown', DATA.series.length, ''],
  ['Window', DATA.meta.lookback_days + ' days', DATA.meta.date_min + ' to ' + DATA.meta.date_max],
  ['Up over window', nGainers + ' / ' + DATA.series.length, ''],
].map(([label,val,sub])=>`<div class="kpi-card"><div class="kpi-label">${label}</div><div class="kpi-value">${val}</div><div class="kpi-label" style="text-transform:none;margin-top:4px;">${sub}</div></div>`).join('');

const grid = document.getElementById('chart-grid');
DATA.series.forEach((s, i) => {
  const div = document.createElement('div');
  div.className = 'chart-container';
  const retClass = s.period_return >= 0 ? 'pos' : 'neg';
  const retStr = (s.period_return>=0?'+':'') + s.period_return.toFixed(1) + '%';
  div.innerHTML = `<h3><span>${s.symbol}</span><span class="${retClass}">${retStr}</span></h3><canvas id="chart-${i}"></canvas>`;
  grid.appendChild(div);
  new Chart(div.querySelector('canvas'), {
    type: 'line',
    data: { labels: s.dates, datasets: [{ data: s.closes, borderColor:'#4f8cff', backgroundColor:'rgba(79,140,255,0.08)', fill:true, borderWidth:1.5, pointRadius:0, tension:0.15 }] },
    options: { responsive:true, animation:false, plugins:{legend:{display:false}},
      scales:{ x:{ticks:{maxTicksLimit:6},grid:{display:false}}, y:{grid:{color:'#232b40'}} } }
  });
});
</script>
</body>
</html>
"""


def build_and_write(out_dir: Path, symbols: list[str], history_csv: Path):
    hist = load_price_history(history_csv, symbols)
    out_path = out_dir / "dashboard_performance_top10.html"

    if hist is None or hist.empty:
        out_path.write_text(
            PLACEHOLDER_HTML.format(sym0=symbols[0] if symbols else "SYMBOL",
                                     symbols=", ".join(symbols)),
            encoding="utf-8",
        )
        print(f"[performance] no history at {history_csv} -- wrote placeholder to {out_path}")
        return

    series = []
    for sym, g in hist.groupby("symbol"):
        g = g.sort_values("date")
        period_return = float(g["close"].iloc[-1] / g["close"].iloc[0] - 1) * 100
        series.append({
            "symbol": sym,
            "dates": g["date"].dt.strftime("%Y-%m-%d").tolist(),
            "closes": g["close"].round(2).tolist(),
            "period_return": round(period_return, 2),
        })
    # keep the caller's rank order where possible
    series.sort(key=lambda s: symbols.index(s["symbol"]) if s["symbol"] in symbols else 999)

    data = {
        "meta": {
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            "lookback_days": LOOKBACK_DAYS,
            "date_min": hist["date"].min().strftime("%Y-%m-%d"),
            "date_max": hist["date"].max().strftime("%Y-%m-%d"),
        },
        "series": series,
    }
    html = HTML_TEMPLATE.replace("__DATA_JSON__", json.dumps(data))
    out_path.write_text(html, encoding="utf-8")
    print(f"[performance] wrote {out_path} ({out_path.stat().st_size/1024:.0f} KB)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out")
    ap.add_argument("--history-csv", default="data/price_history.csv")
    args = ap.parse_args()
    out_dir = Path(args.out)

    full_df = pd.read_csv(out_dir / "nepse_heston_screener_full.csv")
    top_symbols = full_df.sort_values("rank").head(10)["symbol"].tolist()

    build_and_write(out_dir, top_symbols, Path(args.history_csv))


if __name__ == "__main__":
    main()
