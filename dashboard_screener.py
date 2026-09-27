"""
dashboard_screener.py
=======================
Builds a single self-contained HTML dashboard for the top-N Heston/Monte
Carlo screener results: KPI cards, a final-score bar chart, a risk/return
scatter across the WHOLE universe with the top-N highlighted, a per-stock
Heston Monte Carlo fan chart (percentile bands + sample paths) with a
stock picker, a terminal-return histogram, and a full metrics table.

Run after screener.py (reuses its outputs):
    python dashboard_screener.py --out out
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


def build_dashboard_data(out_dir: Path, dynamics_info: dict) -> dict:
    full_df = pd.read_csv(out_dir / "nepse_heston_screener_full.csv")
    npz = np.load(out_dir / "top_n_paths.npz", allow_pickle=True)
    S, symbols, S0 = npz["S"], npz["symbols"], npz["S0"]  # S: (n_top, n_paths, n_days+1)

    top_df = full_df[full_df["symbol"].isin(symbols)].set_index("symbol").loc[symbols].reset_index()

    top_records = []
    for _, r in top_df.iterrows():
        top_records.append({
            "rank": int(r["rank"]), "symbol": r["symbol"], "sector": r["sector"],
            "S0": round(float(r["S0"]), 2),
            "realized_volatility": round(float(r["realized_volatility"]) * 100, 2),
            "mu": round(float(r["mu"]) * 100, 2),
            "mc_expected_return": round(float(r["mc_expected_return"]) * 100, 2),
            "mc_median_return": round(float(r["mc_median_return"]) * 100, 2),
            "mc_prob_gain": round(float(r["mc_prob_gain"]) * 100, 1),
            "mc_var_5": round(float(r["mc_var_5"]) * 100, 2),
            "mc_cvar_5": round(float(r["mc_cvar_5"]) * 100, 2),
            "mc_risk_adj_return": round(float(r["mc_risk_adj_return"]), 3),
            "shrunk_score": round(float(r["shrunk_score"]), 3) if "shrunk_score" in r and pd.notna(r["shrunk_score"]) else None,
            "final_score": round(float(r["final_score"]), 3),
        })

    scatter_all = []
    top_set = set(symbols.tolist())
    for _, r in full_df.iterrows():
        if pd.isna(r.get("mc_expected_return")) or pd.isna(r.get("mc_return_vol")):
            continue
        scatter_all.append({
            "symbol": r["symbol"],
            "x": round(float(r["mc_expected_return"]) * 100, 2),
            "y": round(float(r["mc_return_vol"]) * 100, 2),
            "top": r["symbol"] in top_set,
        })

    n_days = S.shape[2] - 1
    days_axis = list(range(0, n_days + 1))
    rng = np.random.default_rng(11)
    fan = {}
    hist = {}
    for i, sym in enumerate(symbols):
        paths = S[i]  # (n_paths, n_days+1)
        pct = np.percentile(paths, [5, 25, 50, 75, 95], axis=0)
        sample_idx = rng.choice(paths.shape[0], size=min(18, paths.shape[0]), replace=False)
        fan[sym] = {
            "days": days_axis,
            "p5": np.round(pct[0], 2).tolist(),
            "p25": np.round(pct[1], 2).tolist(),
            "median": np.round(pct[2], 2).tolist(),
            "p75": np.round(pct[3], 2).tolist(),
            "p95": np.round(pct[4], 2).tolist(),
            "samples": [np.round(paths[j], 2).tolist() for j in sample_idx],
            "S0": round(float(S0[i]), 2),
        }
        term_ret = (paths[:, -1] / S0[i] - 1.0) * 100
        counts, edges = np.histogram(term_ret, bins=30)
        hist[sym] = {
            "edges": np.round(edges, 2).tolist(),
            "counts": counts.tolist(),
        }

    meta = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "universe_size": int(len(full_df)),
        "horizon_days": int(n_days),
        "n_paths_dashboard": int(S.shape[1]),
        "dynamics": dynamics_info,
    }

    return {"meta": meta, "top10": top_records, "scatter_all": scatter_all, "fan": fan, "hist": hist}


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>NEPSE Heston Screener -- Top 10</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.5.1"></script>
<style>
  :root{
    --bg:#0f1420; --bg-card:#161d2e; --bg-header:#0b0f1a; --border:#232b40;
    --text:#e8ecf6; --text-secondary:#93a0bf; --accent:#4f8cff; --accent2:#22c58b;
    --negative:#ff5c7c; --positive:#22c58b; --radius:12px; --gap:16px;
  }
  *{box-sizing:border-box;}
  body{margin:0;background:var(--bg);color:var(--text);font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;}
  .wrap{max-width:1320px;margin:0 auto;padding:20px;}
  header{background:var(--bg-header);border:1px solid var(--border);border-radius:var(--radius);padding:20px 24px;margin-bottom:var(--gap);display:flex;justify-content:space-between;flex-wrap:wrap;gap:12px;align-items:center;}
  header h1{font-size:20px;margin:0 0 4px 0;}
  header p{margin:0;color:var(--text-secondary);font-size:13px;}
  .badge{display:inline-block;background:#1b2740;border:1px solid var(--border);color:var(--text-secondary);padding:4px 10px;border-radius:20px;font-size:12px;margin-right:6px;}
  .kpi-row{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:var(--gap);margin-bottom:var(--gap);}
  .kpi-card{background:var(--bg-card);border:1px solid var(--border);border-radius:var(--radius);padding:16px 20px;}
  .kpi-label{font-size:12px;color:var(--text-secondary);text-transform:uppercase;letter-spacing:.5px;margin-bottom:6px;}
  .kpi-value{font-size:24px;font-weight:700;}
  .chart-row{display:grid;grid-template-columns:1fr 1fr;gap:var(--gap);margin-bottom:var(--gap);}
  .chart-container{background:var(--bg-card);border:1px solid var(--border);border-radius:var(--radius);padding:18px 20px;}
  .chart-container h3{font-size:14px;margin:0 0 4px 0;}
  .chart-container .sub{font-size:12px;color:var(--text-secondary);margin:0 0 12px 0;}
  .chart-container canvas{max-height:320px;}
  .full-row{grid-column:1 / -1;}
  select{background:#1b2740;color:var(--text);border:1px solid var(--border);border-radius:6px;padding:6px 10px;font-size:13px;}
  table{width:100%;border-collapse:collapse;font-size:12.5px;}
  th,td{padding:8px 10px;text-align:right;border-bottom:1px solid var(--border);white-space:nowrap;}
  th:first-child,td:first-child,th:nth-child(2),td:nth-child(2),th:nth-child(3),td:nth-child(3){text-align:left;}
  th{color:var(--text-secondary);font-weight:600;text-transform:uppercase;font-size:11px;letter-spacing:.4px;cursor:pointer;}
  tbody tr:hover{background:#1b2740;}
  .pos{color:var(--positive);} .neg{color:var(--negative);}
  .methodology{background:var(--bg-card);border:1px solid var(--border);border-radius:var(--radius);padding:16px 20px;font-size:12.5px;color:var(--text-secondary);line-height:1.6;margin-top:var(--gap);}
  .methodology b{color:var(--text);}
  footer{text-align:center;color:var(--text-secondary);font-size:12px;padding:20px 0;}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <div>
      <h1>NEPSE Heston Monte Carlo Screener</h1>
      <p>Top 10 by blended risk-adjusted score &middot; generated <span id="gen-time"></span></p>
    </div>
    <div id="dyn-badges"></div>
  </header>

  <section class="kpi-row" id="kpi-row"></section>

  <section class="chart-row">
    <div class="chart-container">
      <h3>Final composite score -- top 10</h3>
      <p class="sub">40% MC risk-adjusted return + 40% existing fundamentals/technicals composite + 20% MC prob. of gain (all z-scored)</p>
      <canvas id="score-chart"></canvas>
    </div>
    <div class="chart-container">
      <h3>Risk / return -- full universe</h3>
      <p class="sub">Simulated expected return vs. simulated return volatility (180-trading-day horizon). Top 10 highlighted.</p>
      <canvas id="scatter-chart"></canvas>
    </div>
  </section>

  <section class="chart-row">
    <div class="chart-container full-row">
      <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:10px;">
        <div>
          <h3 style="display:inline;">Heston Monte Carlo fan chart</h3>
          <p class="sub" style="display:inline-block;margin-left:8px;">Simulated price paths, percentile bands (5/25/50/75/95%) + 18 sample paths</p>
        </div>
        <select id="stock-picker"></select>
      </div>
      <canvas id="fan-chart" height="90"></canvas>
    </div>
  </section>

  <section class="chart-row">
    <div class="chart-container full-row">
      <h3>Simulated terminal-return distribution -- <span id="hist-symbol"></span></h3>
      <p class="sub">Distribution of simulated returns at the end of the horizon (same stock as the fan chart above)</p>
      <canvas id="hist-chart"></canvas>
    </div>
  </section>

  <section class="chart-container">
    <h3>Top 10 -- full metrics</h3>
    <div style="overflow-x:auto;">
      <table id="metrics-table">
        <thead><tr>
          <th>#</th><th>Symbol</th><th>Sector</th><th>Price</th><th>Realized&nbsp;Vol</th>
          <th>&mu; (drift)</th><th>E[Return]</th><th>Median</th><th>P(gain)</th>
          <th>VaR&nbsp;5%</th><th>CVaR&nbsp;5%</th><th>Risk-adj</th><th>Existing&nbsp;score</th><th>Final&nbsp;score</th>
        </tr></thead>
        <tbody id="metrics-tbody"></tbody>
      </table>
    </div>
  </section>

  <div class="methodology">
    <b>Methodology &amp; assumptions.</b> Universe: NEPSE-listed operating companies with a real EPS (fund units excluded), from your existing screener pipeline.
    Each stock's current variance state and long-run variance (v&#8320; = &theta;) are set from its own <b>realized_volatility.csv</b> value -- real,
    empirically measured volatility, not an assumed figure. Mean-reversion speed (&kappa;), vol-of-vol (&xi;) and the price/variance correlation
    (&rho;) shown in the badges above are a <b>shared, literature-typical, Feller-safe default</b> applied market-wide, because a single trailing-volatility
    number per stock (rather than a full return history) cannot identify stock-specific dynamics -- swap in <code>calibrate_heston_mom()</code>
    per symbol once daily price history is available. Drift (&mu;) is each stock's earnings yield (Graham-style expected-return proxy from your
    existing fundamentals), clipped to &plusmn;30%/60%. Each stock is simulated independently (its own Heston Monte Carlo run) -- see the
    accompanying README for why, and how to extend this to a joint multi-asset run if you want portfolio-level correlation effects.
  </div>

  <footer>NEPSE Heston Monte Carlo Screener &middot; for research/screening use, not investment advice &middot; data as fetched in your pipeline</footer>
</div>

<script>
const DATA = __DATA_JSON__;

function fmtPct(v){ if(v===null||v===undefined||isNaN(v)) return '--'; const s=v>=0?'+':''; return s+v.toFixed(2)+'%'; }
function cls(v){ return v>=0 ? 'pos':'neg'; }

document.getElementById('gen-time').textContent = DATA.meta.generated_at;
document.getElementById('dyn-badges').innerHTML =
  `<span class="badge">universe: ${DATA.meta.universe_size} stocks</span>` +
  `<span class="badge">horizon: ${DATA.meta.horizon_days}d</span>` +
  `<span class="badge">&kappa;=${DATA.meta.dynamics.kappa.toFixed(2)}</span>` +
  `<span class="badge">&xi;=${DATA.meta.dynamics.xi.toFixed(2)}</span>` +
  `<span class="badge">&rho;=${DATA.meta.dynamics.rho.toFixed(2)}</span>` +
  `<span class="badge">${DATA.meta.n_paths_dashboard.toLocaleString()} paths</span>`;

// ---- KPI cards ----
const top1 = DATA.top10[0];
const avgProb = DATA.top10.reduce((a,r)=>a+r.mc_prob_gain,0)/DATA.top10.length;
const avgRet = DATA.top10.reduce((a,r)=>a+r.mc_expected_return,0)/DATA.top10.length;
const kpis = [
  ['Top pick', top1.symbol, top1.sector],
  ['Avg. simulated return (top 10)', fmtPct(avgRet), '180-trading-day horizon'],
  ['Avg. P(gain) (top 10)', avgProb.toFixed(1)+'%', 'share of simulated paths ending up'],
  ['Universe screened', DATA.meta.universe_size, 'operating-company equities'],
];
document.getElementById('kpi-row').innerHTML = kpis.map(([label,val,sub])=>
  `<div class="kpi-card"><div class="kpi-label">${label}</div><div class="kpi-value">${val}</div><div class="kpi-label" style="text-transform:none;margin-top:4px;">${sub||''}</div></div>`
).join('');

// ---- Score bar chart ----
new Chart(document.getElementById('score-chart'), {
  type: 'bar',
  data: {
    labels: DATA.top10.map(r=>r.symbol),
    datasets: [{ label:'Final score', data: DATA.top10.map(r=>r.final_score),
      backgroundColor: DATA.top10.map((_,i)=> i===0 ? '#22c58b' : '#4f8cff') }]
  },
  options: { responsive:true, plugins:{legend:{display:false}}, scales:{ x:{grid:{display:false}}, y:{grid:{color:'#232b40'}} } }
});

// ---- Risk/return scatter ----
const bg = DATA.scatter_all.filter(p=>!p.top);
const hi = DATA.scatter_all.filter(p=>p.top);
new Chart(document.getElementById('scatter-chart'), {
  type: 'scatter',
  data: { datasets: [
    { label:'universe', data: bg.map(p=>({x:p.x,y:p.y,symbol:p.symbol})), backgroundColor:'rgba(147,160,191,0.35)', pointRadius:3 },
    { label:'top 10', data: hi.map(p=>({x:p.x,y:p.y,symbol:p.symbol})), backgroundColor:'#22c58b', pointRadius:5, pointHoverRadius:7 },
  ]},
  options: { responsive:true,
    plugins:{ legend:{display:false}, tooltip:{ callbacks:{ label:(ctx)=> `${ctx.raw.symbol}: E[r]=${ctx.raw.x.toFixed(1)}%, vol=${ctx.raw.y.toFixed(1)}%` } } },
    scales:{ x:{title:{display:true,text:'Simulated expected return (%)'},grid:{color:'#232b40'}}, y:{title:{display:true,text:'Simulated return volatility (%)'},grid:{color:'#232b40'}} }
  }
});

// ---- Fan chart + histogram (stock picker) ----
const picker = document.getElementById('stock-picker');
DATA.top10.forEach(r=> { const o=document.createElement('option'); o.value=r.symbol; o.textContent=`${r.rank}. ${r.symbol}`; picker.appendChild(o); });

let fanChart=null, histChart=null;
function renderStock(sym){
  const f = DATA.fan[sym];
  const h = DATA.hist[sym];
  document.getElementById('hist-symbol').textContent = sym;

  const bandDatasets = [
    { label:'p95', data:f.p95, borderWidth:0, pointRadius:0, fill:false, borderColor:'transparent' },
    { label:'p75-p95', data:f.p75, borderWidth:0, pointRadius:0, fill:'-1', backgroundColor:'rgba(79,140,255,0.12)', borderColor:'transparent' },
    { label:'median', data:f.median, borderWidth:2, pointRadius:0, borderColor:'#4f8cff', fill:false },
    { label:'p25-p75', data:f.p25, borderWidth:0, pointRadius:0, fill:'-2', backgroundColor:'rgba(79,140,255,0.12)', borderColor:'transparent' },
    { label:'p5-p25', data:f.p5, borderWidth:0, pointRadius:0, fill:'-1', backgroundColor:'rgba(79,140,255,0.10)', borderColor:'transparent' },
  ];
  const sampleDatasets = f.samples.map(s => ({ data:s, borderWidth:1, pointRadius:0, borderColor:'rgba(232,236,246,0.12)', fill:false }));

  if (fanChart) fanChart.destroy();
  fanChart = new Chart(document.getElementById('fan-chart'), {
    type:'line',
    data:{ labels:f.days, datasets:[...sampleDatasets, ...bandDatasets] },
    options:{ responsive:true, animation:false,
      plugins:{ legend:{display:false}, tooltip:{ filter:(item)=> item.datasetIndex>=sampleDatasets.length } },
      scales:{ x:{title:{display:true,text:'Trading days ahead'},grid:{color:'#232b40'}}, y:{title:{display:true,text:'Simulated price'},grid:{color:'#232b40'}} } }
  });

  const labels = h.edges.slice(0,-1).map((e,i)=> ((e+h.edges[i+1])/2).toFixed(1)+'%');
  if (histChart) histChart.destroy();
  histChart = new Chart(document.getElementById('hist-chart'), {
    type:'bar',
    data:{ labels, datasets:[{ label:'paths', data:h.counts, backgroundColor:'#4f8cff' }] },
    options:{ responsive:true, plugins:{legend:{display:false}},
      scales:{ x:{title:{display:true,text:'Simulated return at horizon end'},grid:{display:false},ticks:{maxTicksLimit:15}}, y:{grid:{color:'#232b40'}} } }
  });
}
picker.addEventListener('change', ()=> renderStock(picker.value));
renderStock(DATA.top10[0].symbol);

// ---- Table ----
document.getElementById('metrics-tbody').innerHTML = DATA.top10.map(r => `
  <tr>
    <td>${r.rank}</td><td>${r.symbol}</td><td>${r.sector}</td>
    <td>${r.S0.toLocaleString()}</td><td>${r.realized_volatility.toFixed(1)}%</td>
    <td>${r.mu.toFixed(1)}%</td>
    <td class="${cls(r.mc_expected_return)}">${fmtPct(r.mc_expected_return)}</td>
    <td class="${cls(r.mc_median_return)}">${fmtPct(r.mc_median_return)}</td>
    <td>${r.mc_prob_gain.toFixed(1)}%</td>
    <td class="neg">${fmtPct(r.mc_var_5)}</td>
    <td class="neg">${fmtPct(r.mc_cvar_5)}</td>
    <td>${r.mc_risk_adj_return.toFixed(2)}</td>
    <td>${r.shrunk_score!==null ? r.shrunk_score.toFixed(2) : '--'}</td>
    <td><b>${r.final_score.toFixed(2)}</b></td>
  </tr>`).join('');
</script>
</body>
</html>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out")
    args = ap.parse_args()
    out_dir = Path(args.out)

    # Recover the dynamics used, from screener.py's default (kept in sync here;
    # if you change screener.py's dynamics call, mirror it here or pass via file).
    from heston_model import default_dynamics
    full_df = pd.read_csv(out_dir / "nepse_heston_screener_full.csv")
    dyn = default_dynamics(theta=float(full_df["theta"].median()))
    dyn_info = {"kappa": dyn.kappa, "xi": dyn.xi, "rho": dyn.rho, "source": dyn.source}

    data = build_dashboard_data(out_dir, dyn_info)
    html = HTML_TEMPLATE.replace("__DATA_JSON__", json.dumps(data))

    out_path = out_dir / "dashboard_screener_top10.html"
    out_path.write_text(html, encoding="utf-8")
    print(f"[dashboard] wrote {out_path} ({out_path.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
