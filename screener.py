"""
screener.py
============
Main pipeline:
  1. load & merge NEPSE data (data_loader.load_universe)
  2. get shared Heston dynamics (kappa, xi, rho) -- literature default,
     Feller-safe (heston_model.default_dynamics)
  3. run ONE batched Heston Monte Carlo pass covering every stock in
     the universe (heston_model.simulate_heston_batch) -- terminal
     prices only, no full path storage, for speed
  4. turn simulated terminal prices into risk/return metrics per stock
  5. blend those metrics with your EXISTING Bayesian composite
     (shrunk_score) into one final_score, rank, export full CSV
  6. re-simulate just the top N with more paths AND full path storage,
     for the dashboards

Run: python screener.py [--data-dir data] [--out out] [--top-n 10]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from data_loader import load_universe
from heston_model import default_dynamics, simulate_heston_batch, mc_return_metrics

# --------------------------------------------------------------------
# Config -- change these freely
# --------------------------------------------------------------------
HORIZON_DAYS = 180          # forward simulation horizon (trading days) -- mirrors the
                             # 180-day *lookback* window used for the performance dashboard
DT = 1 / 252                 # annualization convention (trading days/year)
N_PATHS_SCREEN = 20_000       # paths for the full-universe ranking pass (terminal only)
N_PATHS_DASHBOARD = 5_000    # paths for the top-N full-path re-simulation (for charts)
SEED = 7                      # set to None for non-reproducible runs
SCORE_WEIGHTS = dict(risk_adj=0.40, existing_composite=0.40, prob_gain=0.20)


def zscore(s: pd.Series) -> pd.Series:
    sd = s.std(ddof=0)
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - s.mean()) / sd


def run_screen(data_dir: str, top_n: int = 10, seed: int | None = SEED):
    universe = load_universe(data_dir)
    dyn = default_dynamics(theta=float(universe["theta"].median()))

    print(f"[screener] universe: {len(universe)} operating-company equities")
    print(f"[screener] shared Heston dynamics -> kappa={dyn.kappa:.3f} xi={dyn.xi:.3f} "
          f"rho={dyn.rho:.3f} (source={dyn.source})")
    print(f"[screener] horizon={HORIZON_DAYS} trading days, paths={N_PATHS_SCREEN:,}")

    S_T, _ = simulate_heston_batch(
        S0=universe["S0"].values, v0=universe["v0"].values, theta=universe["theta"].values,
        mu=universe["mu"].values, kappa=dyn.kappa, xi=dyn.xi, rho=dyn.rho,
        n_days=HORIZON_DAYS, n_paths=N_PATHS_SCREEN, dt=DT, seed=seed, store_paths=False,
    )
    metrics = mc_return_metrics(universe["S0"].values, S_T)
    df = pd.concat([universe.reset_index(drop=True), metrics.reset_index(drop=True)], axis=1)

    # Blend the new Heston-MC risk/return signal with the EXISTING fundamental+
    # technical Bayesian composite you already computed (shrunk_score), so the
    # final ranking reflects both "what does the fundamentals+technicals model
    # say" and "what does simulating real volatility forward say about the
    # risk-adjusted payoff".
    df["z_risk_adj"] = zscore(df["mc_risk_adj_return"])
    df["z_existing_composite"] = zscore(df["shrunk_score"]) if "shrunk_score" in df else 0.0
    df["z_prob_gain"] = zscore(df["mc_prob_gain"])
    df["final_score"] = (
        SCORE_WEIGHTS["risk_adj"] * df["z_risk_adj"]
        + SCORE_WEIGHTS["existing_composite"] * df["z_existing_composite"]
        + SCORE_WEIGHTS["prob_gain"] * df["z_prob_gain"]
    )
    df = df.sort_values("final_score", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", np.arange(1, len(df) + 1))

    top = df.head(top_n).copy()

    # Re-simulate the top N with more paths and FULL path storage for the
    # dashboards (fan charts / terminal distributions). Independent run
    # (different noise draws) from the screening pass -- that's fine, the
    # screening pass already did its job (selecting these names); this pass
    # exists purely to visualize their distribution in more detail.
    print(f"[screener] re-simulating top {top_n} with {N_PATHS_DASHBOARD:,} paths "
          f"(full paths stored) for the dashboard...")
    S_full, v_full = simulate_heston_batch(
        S0=top["S0"].values, v0=top["v0"].values, theta=top["theta"].values,
        mu=top["mu"].values, kappa=dyn.kappa, xi=dyn.xi, rho=dyn.rho,
        n_days=HORIZON_DAYS, n_paths=N_PATHS_DASHBOARD, dt=DT,
        seed=None if seed is None else seed + 1, store_paths=True,
    )

    return dict(full_df=df, top_df=top, dynamics=dyn, S_paths=S_full, v_paths=v_full,
                horizon_days=HORIZON_DAYS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out", default="out")
    ap.add_argument("--top-n", type=int, default=10)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    result = run_screen(args.data_dir, top_n=args.top_n)
    csv_path = out_dir / "nepse_heston_screener_full.csv"
    result["full_df"].to_csv(csv_path, index=False)
    print(f"[screener] wrote {csv_path} ({len(result['full_df'])} rows)")

    print("\nTop", args.top_n, "by final_score:")
    cols = ["rank", "symbol", "sector", "S0", "realized_volatility", "mu",
            "mc_expected_return", "mc_prob_gain", "mc_risk_adj_return",
            "shrunk_score", "final_score"]
    cols = [c for c in cols if c in result["top_df"].columns]
    print(result["top_df"][cols].to_string(index=False))

    # Stash the simulated arrays for the dashboard script to reuse without
    # re-simulating (npz keeps this self-contained, no pickle of code objects)
    np.savez_compressed(
        out_dir / "top_n_paths.npz",
        S=result["S_paths"], v=result["v_paths"],
        symbols=result["top_df"]["symbol"].values, S0=result["top_df"]["S0"].values,
    )
    print(f"[screener] wrote {out_dir / 'top_n_paths.npz'}")


if __name__ == "__main__":
    main()
