"""
build_priors.py - Mathematical Prior Construction for NEPSE Models
===================================================================
Incorporates foundational probability theory and distribution theorems:
  1. Gamma Conjugate Variance Priors (PDF 3, Page 2):
     f(x; alpha, beta) = (beta^alpha / Gamma(alpha)) * x^(alpha - 1) * e^(-beta * x)
     Mean = alpha / beta, Variance = alpha / beta^2, Mode = (alpha - 1) / beta.
     Provides independent control over prior location (realized variance)
     and prior confidence (coefficient of variation), replacing HalfNormal.

  2. Beta Distribution for Leverage Correlation rho in [-1, 1] (PDF 3, Page 3):
     f(u; m, n) = (u^(m - 1) * (1 - u)^(n - 1)) / B(m, n) for u in [0, 1].
     Transforms rho in [-1, 1] via u = (rho + 1) / 2 to a Beta(m, n) distribution
     centered at empirical leverage correlation.

  3. Cauchy-Schwarz Inequality for Variance Bounds (PDF 1, Page 5):
     E(XY) <= sqrt(E(X^2) * E(Y^2)), |Cov(X, Y)| <= sigma_X * sigma_Y.
     Bounds the maximum possible covariance between return shocks and volatility.

  4. Laplace Distribution for Robust Scale & Z-scoring (PDF 3, Page 4):
     f(x) = (1 / (2b)) * exp(-|x - mu| / b).
     MAD-based robust scaling invariant to extreme circuit-breaker jumps (+10% / -10%).
"""

import numpy as np
import pandas as pd


def gamma_prior_params(target_mean: float | None, target_cv: float = 0.50, floor: float = 1e-4) -> tuple[float, float]:
    """
    Computes shape (alpha) and rate (beta) parameters for a Gamma(alpha, beta) prior.

    From Probability Distributions Reference (PDF 3, Page 2):
        E[X] = alpha / beta
        Var(X) = alpha / beta^2
        CV = sqrt(Var) / E[X] = 1 / sqrt(alpha)

    Solving for alpha and beta:
        alpha = 1 / (target_cv^2)
        beta  = alpha / target_mean

    Parameters:
        target_mean: Realized annualized variance (e.g. from realized_volatility.csv^2).
                     If None or invalid, falls back to weakly informative default.
        target_cv: Coefficient of variation (sigma / mu). Default 0.50 corresponds
                   to 50% relative prior uncertainty (alpha = 4.0).
        floor: Minimum variance floor to prevent numerical collapse.

    Returns:
        (alpha, beta): Tuple of floats where beta is the rate parameter (1/scale).
    """
    if target_mean is None or not np.isfinite(target_mean) or target_mean <= 0:
        # Weakly informative fallback centered at ~0.09 (30% annualized vol)
        target_mean = 0.09
        target_cv = 0.80

    mean_val = max(float(target_mean), floor)
    cv_val = max(float(target_cv), 0.10)

    alpha = 1.0 / (cv_val ** 2)
    beta = alpha / mean_val

    return float(alpha), float(beta)


def beta_rho_prior_params(empirical_rho: float | None, confidence: float = 12.0) -> tuple[float, float]:
    """
    Constructs Beta(m, n) prior parameters for Heston leverage parameter rho in [-1, 1].

    From Probability Distributions Reference (PDF 3, Page 3):
        f(u; m, n) = u^(m - 1) * (1 - u)^(n - 1) / B(m, n), u in [0, 1]
        Mean = m / (m + n), Mode = (m - 1) / (m + n - 2)

    The bijection u = (rho + 1) / 2 maps [-1, 1] -> [0, 1].
    For NEPSE, equities exhibit mild to moderate leverage effect (rho < 0) or
    momentum correlation (rho >= 0).

    Parameters:
        empirical_rho: Sample correlation between daily return and proxy variance change.
        confidence: Effective sample size (m + n). Default 12 provides a gentle,
                    well-behaved prior that stabilizes MCMC without overwhelming likelihood.

    Returns:
        (m, n): Alpha and beta shape parameters for Beta distribution on [0, 1].
    """
    if empirical_rho is None or not np.isfinite(empirical_rho):
        # Default mild negative correlation (classical equity leverage: rho ~ -0.25)
        empirical_rho = -0.25

    rho_clipped = float(np.clip(empirical_rho, -0.85, 0.85))
    u_target = (rho_clipped + 1.0) / 2.0  # mapped to (0, 1)

    m = max(u_target * confidence, 1.2)
    n = max((1.0 - u_target) * confidence, 1.2)

    return float(m), float(n)


def cauchy_schwarz_variance_bound(returns: np.ndarray, var_proxy: np.ndarray) -> dict[str, float]:
    """
    Verifies the Cauchy-Schwarz inequality on empirical return shocks and variance shocks.

    From Probability Distributions Reference (PDF 1, Page 5):
        Cauchy-Schwarz: E(XY) <= sqrt(E(X^2) * E(Y^2))
        Covariance Bound: |Cov(X, Y)| <= sigma_X * sigma_Y
        Correlation Bound: |r| = |Cov(X,Y)| / (sigma_X * sigma_Y) <= 1.0

    This bounds the physically plausible range for volatility-of-volatility (xi)
    and leverage (rho) given the discrete sample.
    """
    r = np.asarray(returns, dtype=float)
    v = np.asarray(var_proxy, dtype=float)

    valid = np.isfinite(r) & np.isfinite(v)
    r = r[valid]
    v = v[valid]

    if len(r) < 10:
        return {"cs_ratio": 0.0, "max_cov": 1e-4, "empirical_cov": 0.0, "xi_upper_bound": 2.0}

    r_c = r - np.mean(r)
    v_c = v - np.mean(v)

    sigma_r = float(np.std(r_c, ddof=1))
    sigma_v = float(np.std(v_c, ddof=1))

    max_cov = sigma_r * sigma_v
    empirical_cov = float(np.mean(r_c * v_c))
    cs_ratio = abs(empirical_cov) / (max_cov + 1e-12)

    # Vol-of-vol (xi) upper bound derived from variance process scale
    xi_upper_bound = float(np.clip(sigma_v * np.sqrt(252) * 2.0, 0.20, 5.0))

    return {
        "cs_ratio": min(cs_ratio, 1.0),
        "max_cov": max_cov,
        "empirical_cov": empirical_cov,
        "xi_upper_bound": xi_upper_bound,
    }


def laplace_robust_moments(data: pd.Series | np.ndarray) -> dict[str, float]:
    """
    Computes robust location and scale based on Laplace (Double Exponential) distribution.

    From Probability Distributions Reference (PDF 3, Page 4):
        f(x) = (1 / (2b)) * exp(-|x - mu| / b)
        MLE location: mu = median
        MLE scale: b = mean absolute deviation from median (MAD)
        Variance: 2 * b^2
        Excess Kurtosis: gamma_2 = 3.0 (leptokurtic, fat-tailed)
    """
    clean = pd.Series(data).dropna().values.astype(float)
    if len(clean) == 0:
        return {"median": 0.0, "mad_scale": 1.0, "robust_sigma": 1.0, "kurtosis": 3.0}

    med = float(np.median(clean))
    # Median Absolute Deviation (MAD)
    mad = float(np.median(np.abs(clean - med)))
    if mad < 1e-8:
        mad = float(np.mean(np.abs(clean - med))) or 1.0

    # Under Normal: sigma = 1.4826 * MAD; under Laplace: b = MAD / ln(2) ~ 1.4427 * MAD
    robust_sigma = 1.4826 * mad
    b_scale = mad / np.log(2)

    return {
        "median": med,
        "mad_scale": b_scale,
        "robust_sigma": robust_sigma,
        "kurtosis": 3.0,
    }
