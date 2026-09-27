"""
heston_model.py
================
Core Heston stochastic-volatility engine used by the NEPSE screener.

The Heston model (Heston, 1993) describes a stock price S_t and its
LATENT (unobserved) instantaneous variance v_t jointly:

    dS_t = mu * S_t dt + sqrt(v_t) * S_t dW1_t
    dv_t = kappa * (theta - v_t) dt + xi * sqrt(v_t) dW2_t
    corr(dW1_t, dW2_t) = rho

    mu    : expected (real-world) drift of the stock
    v_t   : latent instantaneous variance (v_t = "spot vol"^2) -- this is
            the thing that is NOT directly observable and that the whole
            model exists to describe
    kappa : speed of mean reversion of variance back to its long-run level
    theta : long-run ("equilibrium") variance
    xi    : volatility-of-volatility (how noisy the variance process is)
    rho   : correlation between price shocks and variance shocks
            (usually negative for equities -- the "leverage effect":
            prices fall, volatility rises)

WHAT "CALIBRATION" NEEDS
-------------------------
A full, per-stock calibration of (kappa, theta, xi, rho) requires a
TIME SERIES of returns for that stock (so you can estimate how fast
variance reverts, how noisy it is, and its correlation with price
moves). `calibrate_heston_mom()` below does this via a discretized
method-of-moments regression, and is ready to use the moment you have
a daily price history per symbol.

Right now (nepse_heston_screener) you only supplied ONE realized-
volatility number per stock (realized_volatility.csv), not a return
series. That number is real, empirical data (not invented), and we
use it honestly as:

    v0    = theta = realized_volatility ** 2

i.e. "assume the stock is currently at its own trailing long-run
variance level" -- the only assumption a single scalar supports.

For (kappa, xi, rho), which describe *dynamics* and cannot be backed
out of a single number, we use `default_dynamics()`: a shared,
literature-typical, Feller-safe parameterization applied market-wide.
This is clearly a modeling assumption, not a calibration, and is kept
separate/visible in the output rather than dressed up as fitted data.
Swap in `calibrate_heston_mom()` per symbol as soon as you have daily
closes (see fetch_historical_prices.py stub).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass


# --------------------------------------------------------------------------
# Dynamics parameters
# --------------------------------------------------------------------------

@dataclass
class HestonDynamics:
    kappa: float   # mean-reversion speed
    xi: float      # vol-of-vol
    rho: float     # price/variance correlation
    source: str    # 'default' | 'market_estimated:<n_obs>' -- provenance, always kept visible


def default_dynamics(theta: float, kappa: float = 3.0, rho: float = -0.5,
                      feller_margin: float = 0.9) -> HestonDynamics:
    """
    Literature-typical equity Heston dynamics, with xi set as a fraction
    of the Feller boundary so the Feller condition (2*kappa*theta > xi^2)
    -- which keeps variance from hitting zero too often -- is always
    satisfied by construction, regardless of a stock's own vol level.

    kappa=3.0  -> variance half-life ~= ln(2)/3 ~= 0.23y (~58 trading days),
                  a common order-of-magnitude for equity variance mean
                  reversion in the absence of stock-specific data.
    rho=-0.5   -> typical equity "leverage effect" (vol rises as price falls).
    feller_margin=0.9 -> xi set to 90% of the Feller boundary: enough
                  vol-of-vol to matter, without letting the discretized
                  variance path misbehave.
    """
    xi = feller_margin * np.sqrt(max(2.0 * kappa * theta, 1e-12))
    return HestonDynamics(kappa=kappa, xi=xi, rho=rho, source="default")


def calibrate_heston_mom(log_returns: np.ndarray, dt: float = 1 / 252,
                          rv_window: int = 20) -> HestonDynamics:
    """
    Method-of-moments Heston calibration from an ACTUAL return series.

    Not used on the current NEPSE upload (no per-symbol return history
    was provided) -- included so this becomes a straight swap-in once
    you have daily closes per symbol (see fetch_historical_prices.py).

    Approach:
      1. Proxy the unobservable instantaneous variance v_t with a
         rolling realized-variance estimator (mean of squared demeaned
         log returns over `rv_window` days, annualized). This is the
         standard practical stand-in for the Heston latent variance
         absent an options-implied surface or a particle filter.
      2. Discretize dv = kappa*theta*dt - kappa*v_t*dt + noise and fit
         by OLS: dv_t = a + b*v_t  =>  kappa = -b/dt, theta = a/(kappa*dt).
      3. xi from the residual variance of that regression, scaled by
         the model's own v_t*dt term.
      4. rho from the correlation between the return innovations and
         the variance innovations (the empirical leverage effect).
    """
    r = np.asarray(log_returns, dtype=float)
    r = r[~np.isnan(r)]
    if len(r) < rv_window + 5:
        raise ValueError(f"Need at least {rv_window + 5} return observations, got {len(r)}")

    r_dem = r - r.mean()
    rv = pd.Series(r_dem ** 2).rolling(rv_window).mean().values / dt  # annualized variance proxy
    valid = ~np.isnan(rv)
    rv = rv[valid]
    r_aligned = r[valid]

    v_t, v_tp1 = rv[:-1], rv[1:]
    dv = v_tp1 - v_t

    X = np.column_stack([np.ones_like(v_t), v_t])
    coef, *_ = np.linalg.lstsq(X, dv, rcond=None)
    a, b = coef
    kappa = max(-b / dt, 1e-4)
    theta = max(a / (kappa * dt), 1e-8)

    resid = dv - (a + b * v_t)
    denom = np.maximum(v_t * dt, 1e-12)
    xi = float(np.sqrt(np.mean(resid ** 2 / denom)))
    xi = min(xi, 0.95 * np.sqrt(2 * kappa * theta))  # keep Feller-safe

    r_for_corr = r_aligned[1:len(dv) + 1]
    n = min(len(r_for_corr), len(dv))
    rho = float(np.corrcoef(r_for_corr[:n], dv[:n])[0, 1]) if n > 3 else -0.5
    rho = float(np.clip(rho, -0.95, 0.95))

    kappa = float(np.clip(kappa, 0.5, 10.0))
    return HestonDynamics(kappa=kappa, xi=xi, rho=rho, source=f"market_estimated:n={len(r)}")


# --------------------------------------------------------------------------
# Monte Carlo engine
# --------------------------------------------------------------------------

def simulate_heston_batch(S0: np.ndarray, v0: np.ndarray, theta: np.ndarray, mu: np.ndarray,
                           kappa: float, xi: float, rho: float,
                           n_days: int, n_paths: int, dt: float = 1 / 252,
                           seed: int | None = None, store_paths: bool = False):
    """
    Vectorized Heston Monte Carlo for MANY symbols at once, using a
    SHARED (kappa, xi, rho) but per-symbol (S0, v0, theta, mu).

    Why batch across symbols instead of a Python loop per symbol?
    kappa/xi/rho are shared market-wide assumptions here (see module
    docstring), so every symbol's simulation differs only through its
    own starting price/variance/drift -- exactly what numpy broadcasting
    is for. This runs ALL symbols' paths in one pass over `n_days`
    instead of one pass per symbol, with identical math either way.
    Each symbol still gets its own independent random draws (no
    cross-stock correlation is assumed -- see README for when you'd
    want that instead).

    Scheme: log-Euler for price, "full truncation" Euler (Lord et al.,
    2010) for variance -- the standard, simple-to-implement discretization
    that keeps variance numerically well-behaved without needing exact
    (Broadie-Kaya) simulation.

    Shapes: S0, v0, theta, mu are 1-D arrays of length n_symbols.
    Returns terminal (S_T, v_T), each shape (n_symbols, n_paths); if
    store_paths, also returns full S, v arrays of shape
    (n_symbols, n_paths, n_days+1).
    """
    rng = np.random.default_rng(seed)
    n_symbols = len(S0)
    S0 = np.asarray(S0, dtype=float).reshape(-1, 1)
    v0 = np.asarray(v0, dtype=float).reshape(-1, 1)
    theta = np.asarray(theta, dtype=float).reshape(-1, 1)
    mu = np.asarray(mu, dtype=float).reshape(-1, 1)

    shape = (n_symbols, n_paths)
    if store_paths:
        S = np.empty((n_symbols, n_paths, n_days + 1))
        v = np.empty((n_symbols, n_paths, n_days + 1))
        S[:, :, 0] = S0
        v[:, :, 0] = v0
    else:
        S_cur = np.broadcast_to(S0, shape).copy()
        v_cur = np.broadcast_to(v0, shape).copy()

    for t in range(n_days):
        z1 = rng.standard_normal(shape)
        z2 = rng.standard_normal(shape)
        w1 = z1
        w2 = rho * z1 + np.sqrt(max(1 - rho ** 2, 0.0)) * z2

        v_now = v[:, :, t] if store_paths else v_cur
        S_now = S[:, :, t] if store_paths else S_cur

        v_pos = np.maximum(v_now, 0.0)
        S_next = S_now * np.exp((mu - 0.5 * v_pos) * dt + np.sqrt(v_pos * dt) * w1)
        v_next = v_now + kappa * (theta - v_pos) * dt + xi * np.sqrt(v_pos * dt) * w2

        if store_paths:
            S[:, :, t + 1] = S_next
            v[:, :, t + 1] = v_next
        else:
            S_cur, v_cur = S_next, v_next

    if store_paths:
        return S, v
    return S_cur, v_cur


def mc_return_metrics(S0: np.ndarray, S_T: np.ndarray) -> pd.DataFrame:
    """
    Turn simulated terminal prices into per-symbol screener metrics.
    S0: (n_symbols,)   S_T: (n_symbols, n_paths)
    """
    S0 = np.asarray(S0, dtype=float).reshape(-1, 1)
    rets = S_T / S0 - 1.0

    var5 = np.percentile(rets, 5, axis=1)
    cvar5 = np.array([rets[i, rets[i] <= var5[i]].mean() for i in range(rets.shape[0])])

    out = pd.DataFrame({
        "mc_expected_return": rets.mean(axis=1),
        "mc_median_return": np.median(rets, axis=1),
        "mc_return_vol": rets.std(axis=1),
        "mc_prob_gain": (rets > 0).mean(axis=1),
        "mc_var_5": var5,
        "mc_cvar_5": cvar5,
        "mc_p25": np.percentile(rets, 25, axis=1),
        "mc_p75": np.percentile(rets, 75, axis=1),
    })
    out["mc_risk_adj_return"] = out["mc_expected_return"] / out["mc_return_vol"].replace(0, np.nan)
    return out
