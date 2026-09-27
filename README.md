# NEPSE Heston Monte Carlo Screener

Turns your existing screener exports into a Heston-stochastic-volatility
Monte Carlo screener, with a results CSV (all stocks) and an HTML
dashboard (top 10).

## Files

| File | Purpose |
|---|---|
| `heston_model.py` | Heston SDE Monte Carlo engine + calibration helpers |
| `data_loader.py` | Loads/merges the NEPSE CSVs into one modeling table |
| `screener.py` | Runs the simulation for the whole universe, scores, ranks, exports CSV |
| `dashboard_screener.py` | Builds `dashboard_screener_top10.html` (data + calculations + charts) |
| `dashboard_performance.py` | Builds `dashboard_performance_top10.html` (180-day actual daily close) |
| `run_all.py` | Runs all of the above in order |

## Quick start

1. Put your 8 CSVs (`fundamentals.csv`, `fundamentals_base.csv`,
   `realized_volatility.csv`, `screener_output.csv`, `sector_ratios.csv`,
   `securities_reference.csv`, `today_price.csv`, `nepse_fuzzy_screener.csv`)
   in a `data/` folder next to these scripts (only `screener_output.csv`
   and `realized_volatility.csv` are actually required by name; the rest
   are optional/for future use).
2. Run:

```bash
pip install numpy pandas
python run_all.py --data-dir data --out out --top-n 10
```

Outputs land in `out/`:
- `nepse_heston_screener_full.csv` -- every screened stock, ranked
- `dashboard_screener_top10.html` -- top-10 dashboard (open in a browser)
- `dashboard_performance_top10.html` -- 180-day price-history dashboard
  (placeholder until you supply history -- see below)

## The Heston model, in this pipeline

```
dS_t = mu * S_t dt + sqrt(v_t) S_t dW1_t
dv_t = kappa (theta - v_t) dt + xi sqrt(v_t) dW2_t,   corr(dW1,dW2) = rho
```

`v_t` is the **latent** variance -- unobservable, which is the whole
point of using Heston instead of assuming a flat/constant volatility
(Black-Scholes/GBM). Per stock, the inputs are:

- **v0 = theta = realized_volatility^2** -- from your `realized_volatility.csv`,
  i.e. real, empirically measured historical volatility per stock, not an
  assumed number. This is the "actual volatility, not fake" part.
  It's used as BOTH the current variance state and the long-run mean,
  because a single trailing-vol scalar is the only thing it can honestly
  support -- it doesn't tell you whether that stock's vol is currently
  above or below its own long-run level, only what the level itself is.
- **mu (drift)** -- each stock's earnings yield (already computed in your
  `screener_output.csv`), a standard Graham-style expected-return proxy,
  clipped to [-30%, +60%] so PE/EPS outliers can't dominate.
- **kappa, xi, rho (dynamics)** -- a single **shared, literature-typical,
  Feller-safe default** (`heston_model.default_dynamics`) applied across
  every stock. This is the one real limitation to be upfront about: a
  single realized-vol number per stock cannot identify how fast that
  stock's OWN variance mean-reverts, how noisy its own vol-of-vol is, or
  its own price/vol correlation -- that requires a return time series.
  `heston_model.calibrate_heston_mom()` is a ready-to-use method-of-moments
  estimator for exactly that, the moment you have daily closes per symbol
  (see "Upgrading to full per-stock calibration" below).

## Should you run one Monte Carlo per stock, or something else?

**One simulation per stock**, yes -- and that's what `screener.py` does
(vectorized so it's one batched *computation*, not one slow Python loop,
but each stock's paths are still independent of every other stock's).
Reasons:

1. Every stock has its own price level, its own volatility (v0/theta),
   and its own drift (mu) -- pooling them into a single simulation would
   average away exactly the differences you're trying to screen on.
2. Nothing here assumes stocks move together. If you DO want that later
   (e.g. total portfolio VaR across a basket, not just per-stock ranking),
   that needs a shared correlation matrix across stocks (Cholesky-decomposed,
   same technique your notes mention you've already built) applied to the
   `z1` draws across symbols at each time step -- a natural extension of
   `simulate_heston_batch`, but a different question from screening.

Compute strategy used here: the full 274-stock universe is simulated once
at 20,000 paths (terminal prices only, for speed) to rank everyone: then
just the top 10 are RE-simulated at 5,000 paths with full path storage
(needed for the fan chart / histograms). Full run: ~70s on this machine.

## Screener scoring

```
final_score = 0.40 * z(mc_risk_adjusted_return)      # E[return] / vol of simulated return
            + 0.40 * z(shrunk_score)                  # your EXISTING Bayesian fundamentals+technicals composite
            + 0.20 * z(mc_probability_of_gain)
```

`z(...)` = cross-sectional z-score. Weights are constants at the top of
`screener.py` (`SCORE_WEIGHTS`) -- change freely. This blends the new
Heston/Monte Carlo risk-adjusted signal with the composite you already
built, rather than replacing it.

## The two dashboards

- **`dashboard_screener_top10.html`** -- fully real, generated from your
  actual uploaded data: KPI cards, final-score bar chart, universe-wide
  risk/return scatter with the top 10 highlighted, a per-stock Heston
  Monte Carlo fan chart (percentile bands + sample paths) with a stock
  picker, terminal-return histogram, and the full metrics table.
- **`dashboard_performance_top10.html`** -- **needs data you don't currently
  have uploaded here**: a daily-close time series per symbol. None of the
  8 files provided contain one (only a single latest price + one
  realized-vol scalar per stock). Rather than fabricate 180 days of
  invented closing prices, `dashboard_performance.py` writes a plain
  placeholder explaining what's missing until you point `--history-csv`
  at a real file shaped:

  ```
  symbol,date,close
  NLO,2026-03-31,238.0
  NLO,2026-04-01,241.5
  ...
  ```

  Your existing NepseScraper-based historical fetch (used previously for
  realized-vol/technical-indicator work) should already be able to emit
  this shape -- once it does, `run_all.py` wires it in automatically.

## Upgrading to full per-stock calibration

Once you have daily closes per symbol:

```python
from heston_model import calibrate_heston_mom
import numpy as np

log_returns = np.diff(np.log(prices))          # prices: daily closes, one symbol
dyn = calibrate_heston_mom(log_returns)         # kappa, xi, rho FOR THIS STOCK
```

Then in `screener.py`, instead of one shared `dyn` for all symbols, loop
per symbol with its own calibrated `dyn` (the `simulate_heston_batch`
signature takes scalar kappa/xi/rho by design for the shared case --
for per-symbol dynamics, call it once per symbol, or extend it to accept
per-symbol arrays the same way S0/v0/theta/mu already are).
