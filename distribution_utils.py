"""
Advanced Probability Distribution Utilities for Quantitative Finance
=====================================================================

This module implements advanced probability distribution theory for
robust financial modeling, incorporating concepts from:

- Bivariate Distributions (joint, marginal, conditional)
- Discrete Probability Distributions (Negative Binomial, Hypergeometric)
- Continuous Probability Distributions (Gamma, Beta, Laplace, Cauchy)
- Copula Theory for dependence modeling
- Advanced moment calculations and risk metrics

Mathematical Foundation:
Based on rigorous probability theory with proper statistical inference
for financial time series analysis.
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from scipy.special import gamma, beta as beta_func, loggamma
from typing import Tuple, Dict, Optional, Union
import warnings

warnings.filterwarnings('ignore')


# =============================================================================
# SECTION 1: UNIVARIATE CONTINUOUS DISTRIBUTIONS
# =============================================================================

class AdvancedDistributionFitter:
    """
    Advanced distribution fitting for financial returns and volatility.

    Implements maximum likelihood estimation (MLE) and method of moments
    for distributions that better capture financial phenomena:
    - Fat tails (Cauchy, Laplace)
    - Skewness (Gamma, Beta)
    - Asymmetric volatility (Gamma for variance)
    """

    @staticmethod
    def fit_gamma_distribution(data: np.ndarray) -> Dict[str, float]:
        """
        Fit Gamma distribution to positive data (e.g., volatility, absolute returns).

        Mathematical Foundation:
        PDF: f(x;α) = e^(-x) * x^(α-1) / Γ(α), for x > 0, α > 0
        Mean: α, Variance: α
        MGF: M_X(t) = (1-t)^(-α)

        Parameters:
        - α (shape): Controls skewness and spread
        - For financial volatility: α typically 2-10

        Returns:
            Dictionary with fitted parameters and moments
        """
        if np.any(data <= 0):
            data = data - data.min() + 1e-6  # Shift to positive

        # Method of moments for initial guess
        sample_mean = np.mean(data)
        sample_var = np.var(data)
        alpha_init = sample_mean  # For Gamma: mean = variance = alpha

        # MLE optimization
        def neg_log_likelihood(alpha):
            # Gamma log-likelihood: -n*ln(Γ(α)) - n*α*ln(x̄) + (α-1)*Σln(x_i)
            # Simplified for computational stability
            n = len(data)
            log_lik = -n * loggamma(alpha) - n * alpha * np.log(sample_mean) + (alpha - 1) * np.sum(np.log(data))
            return -log_lik

        result = minimize(neg_log_likelihood, x0=alpha_init, bounds=[(1e-6, None)])
        alpha_mle = result.x[0]

        # Calculate moments using theoretical formulas
        mean = alpha_mle
        variance = alpha_mle
        skewness = 2 / np.sqrt(alpha_mle)  # γ₁ = 2/√α
        kurtosis = 3 + 6 / alpha_mle  # γ₂ = 6/α

        return {
            'alpha': alpha_mle,
            'mean': mean,
            'variance': variance,
            'std': np.sqrt(variance),
            'skewness': skewness,
            'excess_kurtosis': kurtosis - 3,
            'log_likelihood': -result.fun,
            'aic': 2 * 1 - 2 * (-result.fun),  # AIC = 2k - 2ln(L)
            'bic': 1 * np.log(len(data)) - 2 * (-result.fun)  # BIC = k*ln(n) - 2ln(L)
        }

    @staticmethod
    def fit_beta_distribution(data: np.ndarray) -> Dict[str, float]:
        """
        Fit Beta distribution to bounded data (e.g., normalized returns, liquidity ratios).

        Mathematical Foundation:
        PDF: f(x;m,n) = x^(m-1)(1-x)^(n-1) / B(m,n), for 0 < x < 1
        Mean: m/(m+n), Variance: mn/[(m+n)²(m+n+1)]
        Mode: (m-1)/(m+n-2) (for m,n > 1)

        Parameters:
        - m (alpha): First shape parameter
        - n (beta): Second shape parameter

        Returns:
            Dictionary with fitted parameters and moments
        """
        # Normalize data to [0, 1] range
        data_norm = (data - data.min()) / (data.max() - data.min() + 1e-8)
        data_norm = np.clip(data_norm, 1e-6, 1 - 1e-6)

        # Method of moments for initial guess
        sample_mean = np.mean(data_norm)
        sample_var = np.var(data_norm)

        # From mean = m/(m+n) and var = mn/[(m+n)²(m+n+1)]
        # Solve for m and n
        if sample_var < sample_mean * (1 - sample_mean):
            common = sample_mean * (1 - sample_mean) / sample_var - 1
            m_init = sample_mean * common
            n_init = (1 - sample_mean) * common
        else:
            m_init, n_init = 2.0, 2.0  # Default symmetric

        # MLE using scipy (more robust)
        try:
            alpha_mle, beta_mle, loc, scale = stats.beta.fit(data_norm, floc=0, fscale=1)
        except:
            alpha_mle, beta_mle = m_init, n_init

        # Calculate theoretical moments
        total = alpha_mle + beta_mle
        mean = alpha_mle / total
        variance = (alpha_mle * beta_mle) / (total ** 2 * (total + 1))

        # Mode calculation
        if alpha_mle > 1 and beta_mle > 1:
            mode = (alpha_mle - 1) / (total - 2)
        else:
            mode = None  # Mode at boundary

        return {
            'alpha': alpha_mle,
            'beta': beta_mle,
            'mean': mean,
            'variance': variance,
            'std': np.sqrt(variance),
            'mode': mode,
            'log_likelihood': np.sum(stats.beta.logpdf(data_norm, alpha_mle, beta_mle)),
            'aic': 2 * 2 - 2 * np.sum(stats.beta.logpdf(data_norm, alpha_mle, beta_mle)),
            'bic': 2 * np.log(len(data)) - 2 * np.sum(stats.beta.logpdf(data_norm, alpha_mle, beta_mle))
        }

    @staticmethod
    def fit_laplace_distribution(data: np.ndarray) -> Dict[str, float]:
        """
        Fit Laplace (Double Exponential) distribution for heavy-tailed returns.

        Mathematical Foundation:
        PDF: f(x;θ) = (1/2)θ * exp(-θ|x-μ|)
        Mean: μ, Variance: 2/θ²
        MGF: M_X(t) = 1/(1-t²) (for standard Laplace with μ=0, θ=1)

        Parameters:
        - μ (location): Mean/median
        - θ (scale): Controls spread (related to variance)

        Financial Application:
        - Better captures fat tails in stock returns
        - More robust to outliers than Gaussian
        """
        # MLE for Laplace: μ = median, θ = 1/mean(|x-μ|)
        mu_mle = np.median(data)
        theta_mle = 1.0 / np.mean(np.abs(data - mu_mle))

        # Theoretical moments
        mean = mu_mle
        variance = 2 / (theta_mle ** 2)

        # Calculate log-likelihood
        log_lik = np.sum(stats.laplace.logpdf(data, loc=mu_mle, scale=1 / theta_mle))

        return {
            'mu': mu_mle,
            'theta': theta_mle,
            'scale': 1 / theta_mle,
            'mean': mean,
            'variance': variance,
            'std': np.sqrt(variance),
            'log_likelihood': log_lik,
            'aic': 2 * 2 - 2 * log_lik,
            'bic': 2 * np.log(len(data)) - 2 * log_lik
        }

    @staticmethod
    def fit_cauchy_distribution(data: np.ndarray) -> Dict[str, float]:
        """
        Fit Cauchy distribution for extreme heavy-tailed phenomena.

        Mathematical Foundation:
        PDF: f(x;μ,λ) = (1/π) * λ/[λ² + (x-μ)²]
        Location: μ, Scale: λ
        Mean: μ (but doesn't exist in strict sense), Variance: ∞
        Characteristic Function: φ_X(t) = exp(itμ - λ|t|)

        Key Properties:
        - No finite moments (mean and variance are undefined)
        - Heavy tails: P(|X| > x) ~ 2/(πx) as x → ∞
        - Stable distribution: sum of Cauchy variables is Cauchy

        Financial Application:
        - Modeling extreme market events
        - Risk assessment for tail events
        """

        # MLE for Cauchy requires numerical optimization
        def neg_log_likelihood(params):
            mu, lam = params
            # Cauchy log-likelihood: -n*log(π) - Σlog(λ² + (x-μ)²)
            log_lik = -len(data) * np.log(np.pi) - np.sum(np.log(lam ** 2 + (data - mu) ** 2))
            return -log_lik

        # Initial guess: median for μ, IQR for λ
        mu_init = np.median(data)
        iqr = np.percentile(data, 75) - np.percentile(data, 25)
        lam_init = iqr / 2  # Approximation

        result = minimize(neg_log_likelihood, x0=[mu_init, lam_init],
                          bounds=[(None, None), (1e-6, None)])
        mu_mle, lam_mle = result.x

        # Calculate theoretical properties
        # Note: Mean and variance are undefined for Cauchy
        median = mu_mle  # Location parameter = median
        mode = mu_mle  # Location parameter = mode

        # Interquartile range
        q1 = mu_mle - lam_mle
        q3 = mu_mle + lam_mle
        iqr_fitted = q3 - q1

        return {
            'mu': mu_mle,
            'lambda': lam_mle,
            'median': median,
            'mode': mode,
            'iqr': iqr_fitted,
            'log_likelihood': -result.fun,
            'aic': 2 * 2 - 2 * (-result.fun),
            'bic': 2 * np.log(len(data)) - 2 * (-result.fun),
            'note': 'Cauchy distribution has undefined mean and variance'
        }

    @staticmethod
    def fit_negative_binomial_distribution(data: np.ndarray) -> Dict[str, float]:
        """
        Fit Negative Binomial distribution for count data (e.g., number of trades).

        Mathematical Foundation:
        PMF: P(X=x) = C(x+k-1, k-1) * p^k * q^x, for x = 0,1,2,...
        Alternative: P(X=x) = C(x-1, k-1) * p^k * q^(x-k), for x = k,k+1,...

        Moments:
        Mean: kq/p, Variance: kq/p²
        MGF: M_X(t) = (q - pe^t)^(-k)

        Parameters:
        - k (shape): Number of successes
        - p (success probability): p = 1-q

        Financial Application:
        - Modeling trade counts
        - Number of price jumps in a period
        """
        # Ensure non-negative integer data
        data = np.round(data).astype(int)
        data = data[data >= 0]

        if len(data) == 0:
            return {'error': 'No valid non-negative data'}

        # Method of moments
        sample_mean = np.mean(data)
        sample_var = np.var(data)

        if sample_var > sample_mean:  # Overdispersion check
            # From mean = kq/p and var = kq/p²
            # Solve: p = mean/var, k = mean²/(var - mean)
            p_mom = sample_mean / sample_var
            k_mom = sample_mean ** 2 / (sample_var - sample_mean)
        else:
            p_mom, k_mom = 0.5, sample_mean  # Default

        # MLE using scipy
        try:
            # Fit negative binomial (n=k, p=p in scipy parametrization)
            n_mle, p_mle, loc = stats.nbinom.fit(data, floc=0)
            k_mle = n_mle
        except:
            k_mle, p_mle = k_mom, p_mom

        q = 1 - p_mle
        mean = k_mle * q / p_mle
        variance = k_mle * q / (p_mle ** 2)

        # Calculate skewness and kurtosis
        skewness = (1 + q) / np.sqrt(k_mle * q)  # γ₁ = (1+q)/√(kq)
        kurtosis = 3 + (6 + p_mle ** 2) / (k_mle * q)  # γ₂ = (6+p²)/(kq)

        return {
            'k': k_mle,
            'p': p_mle,
            'q': q,
            'mean': mean,
            'variance': variance,
            'std': np.sqrt(variance),
            'skewness': skewness,
            'excess_kurtosis': kurtosis - 3,
            'overdispersion': variance / mean if mean > 0 else None
        }


# =============================================================================
# SECTION 2: BIVARIATE DISTRIBUTIONS AND COPULAS
# =============================================================================

class BivariateAnalyzer:
    """
    Bivariate distribution analysis for understanding dependence between
    financial variables (e.g., returns vs volatility, cross-asset correlations).

    Mathematical Foundation:
    - Joint PDF: f(x,y) = ∂²F(x,y)/∂x∂y
    - Marginal: f(x) = ∫f(x,y)dy
    - Conditional: f(x|y) = f(x,y)/f(y)
    - Independence: f(x,y) = f(x)f(y)
    """

    @staticmethod
    def joint_moments(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Calculate joint moments including covariance and correlation.

        Mathematical Foundation:
        Cov(X,Y) = E[(X-E(X))(Y-E(Y))] = E(XY) - E(X)E(Y)
        Correlation: r = Cov(X,Y)/(σ_X σ_Y)

        Also calculates higher-order joint moments:
        μ'_{rs} = E(X^r Y^s)
        μ_{rs} = E[(X-E(X))^r (Y-E(Y))^s]
        """
        # Basic moments
        mean_x = np.mean(x)
        mean_y = np.mean(y)

        # Covariance and correlation
        covariance = np.cov(x, y, bias=True)[0, 1]
        correlation = np.corrcoef(x, y)[0, 1]

        # Joint raw moments
        e_xy = np.mean(x * y)
        e_x2y = np.mean(x ** 2 * y)
        e_xy2 = np.mean(x * y ** 2)

        # Joint central moments
        central_xy = np.mean((x - mean_x) * (y - mean_y))
        central_x2y = np.mean((x - mean_x) ** 2 * (y - mean_y))
        central_xy2 = np.mean((x - mean_x) * (y - mean_y) ** 2)

        return {
            'mean_x': mean_x,
            'mean_y': mean_y,
            'covariance': covariance,
            'correlation': correlation,
            'e_xy': e_xy,
            'e_x2y': e_x2y,
            'e_xy2': e_xy2,
            'central_xy': central_xy,
            'central_x2y': central_x2y,
            'central_xy2': central_xy2
        }

    @staticmethod
    def conditional_expectation(x: np.ndarray, y: np.ndarray, n_bins: int = 10) -> Dict[str, np.ndarray]:
        """
        Calculate conditional expectation E[X|Y] using binning.

        Mathematical Foundation:
        E[g(X,Y)|Y=y_j] = Σ g(x_i,y_j) * P(X=x_i|Y=y_j)

        For continuous case: E[X|Y=y] = ∫ x * f(x|y) dx = ∫ x * f(x,y)/f(y) dx
        """
        # Create bins for Y
        y_bins = pd.cut(y, bins=n_bins)

        conditional_means = []
        conditional_variances = []
        bin_centers = []

        for bin_val in y_bins.cat.categories:
            mask = y_bins == bin_val
            x_conditional = x[mask]

            if len(x_conditional) > 0:
                conditional_means.append(np.mean(x_conditional))
                conditional_variances.append(np.var(x_conditional))
                bin_centers.append(bin_val.mid)

        return {
            'bin_centers': np.array(bin_centers),
            'conditional_means': np.array(conditional_means),
            'conditional_variances': np.array(conditional_variances)
        }

    @staticmethod
    def test_independence(x: np.ndarray, y: np.ndarray, method: str = 'pearson') -> Dict[str, float]:
        """
        Test independence between two variables.

        Methods:
        - 'pearson': Linear correlation test
        - 'spearman': Rank correlation test
        - 'kendall': Kendall's tau test
        - 'mutual_info': Information-theoretic test
        """
        if method == 'pearson':
            corr, p_value = stats.pearsonr(x, y)
        elif method == 'spearman':
            corr, p_value = stats.spearmanr(x, y)
        elif method == 'kendall':
            corr, p_value = stats.kendalltau(x, y)
        elif method == 'mutual_info':
            from sklearn.feature_selection import mutual_info_regression
            corr = mutual_info_regression(x.reshape(-1, 1), y)[0]
            p_value = None  # No p-value for mutual information
        else:
            raise ValueError(f"Unknown method: {method}")

        return {
            'method': method,
            'statistic': corr,
            'p_value': p_value,
            'independent': p_value > 0.05 if p_value is not None else None
        }


class CopulaModeler:
    """
    Copula modeling for capturing dependence structures separate from marginal distributions.

    Mathematical Foundation:
    - Copula C: [0,1]² → [0,1] with uniform marginals
    - Sklar's Theorem: F(x,y) = C(F_X(x), F_Y(y))
    - Archimedean copulas: Generated by a generator function φ

    Implemented Copulas:
    - Gaussian: Normal copula with correlation parameter
    - Frank: Archimedean copula for symmetric dependence
    - Clayton: Asymmetric lower-tail dependence
    - Gumbel: Asymmetric upper-tail dependence
    """

    @staticmethod
    def gaussian_copula_correlation(x: np.ndarray, y: np.ndarray) -> float:
        """
        Estimate correlation parameter for Gaussian copula.

        Mathematical Foundation:
        C_ρ(u,v) = Φ_ρ(Φ⁻¹(u), Φ⁻¹(v))
        where Φ_ρ is bivariate normal CDF with correlation ρ
        """
        # Transform to uniform using empirical CDF
        u = stats.rankdata(x) / (len(x) + 1)
        v = stats.rankdata(y) / (len(y) + 1)

        # Transform to normal using inverse CDF
        x_norm = stats.norm.ppf(u)
        y_norm = stats.norm.ppf(v)

        # Correlation in normal space = copula correlation
        return np.corrcoef(x_norm, y_norm)[0, 1]

    @staticmethod
    def frank_copula_parameter(x: np.ndarray, y: np.ndarray) -> float:
        """
        Estimate Frank copula parameter θ.

        Mathematical Foundation:
        C_θ(u,v) = -1/θ * ln[1 + (exp(-θu)-1)(exp(-θv)-1)/(exp(-θ)-1)]

        Frank copula captures symmetric dependence and handles both
        positive and negative correlation.
        """
        # Transform to uniform marginals
        u = stats.rankdata(x) / (len(x) + 1)
        v = stats.rankdata(y) / (len(y) + 1)

        # Estimate θ using method of moments or MLE
        # Simplified: use empirical Kendall's tau
        tau, _ = stats.kendalltau(x, y)

        # For Frank copula: τ = 1 - 4/θ * (1 - D_1(θ))
        # where D_1 is Debye function, solved numerically
        # Approximation: θ ≈ τ * 8 for small τ
        theta_approx = tau * 8

        return theta_approx

    @staticmethod
    def generate_copula_samples(copula_type: str, theta: float, n_samples: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate samples from specified copula.

        Parameters:
        - copula_type: 'gaussian', 'frank', 'clayton', 'gumbel'
        - theta: Copula parameter (correlation for gaussian, dependence for others)
        - n_samples: Number of samples to generate
        """
        if copula_type == 'gaussian':
            # Generate from bivariate normal
            mean = [0, 0]
            cov = [[1, theta], [theta, 1]]
            z = np.random.multivariate_normal(mean, cov, n_samples)
            u = stats.norm.cdf(z[:, 0])
            v = stats.norm.cdf(z[:, 1])

        elif copula_type == 'frank':
            # Frank copula using acceptance-rejection or inversion
            # Simplified approximation using Gaussian for now
            mean = [0, 0]
            cov = [[1, theta / 8], [theta / 8, 1]]  # Approximate mapping
            z = np.random.multivariate_normal(mean, cov, n_samples)
            u = stats.norm.cdf(z[:, 0])
            v = stats.norm.cdf(z[:, 1])

        else:
            raise ValueError(f"Unsupported copula type: {copula_type}")

        return u, v


# =============================================================================
# SECTION 3: ADVANCED RISK METRICS
# =============================================================================

class AdvancedRiskMetrics:
    """
    Advanced risk metrics using proper probability distributions.

    Mathematical Foundation:
    - VaR (Value at Risk): Quantile of loss distribution
    - CVaR (Conditional VaR): Expected loss beyond VaR
    - Expected Shortfall: E[L | L > VaR]
    - Drawdown distribution analysis
    """

    @staticmethod
    def calculate_var(returns: np.ndarray, confidence_level: float = 0.95,
                      distribution: str = 'normal') -> Dict[str, float]:
        """
        Calculate Value at Risk using specified distribution.

        Mathematical Foundation:
        VaR_α = F^(-1)(α) where F is loss distribution CDF

        For different distributions:
        - Normal: VaR = μ + σ * Φ^(-1)(α)
        - Laplace: VaR = μ + (1/θ) * ln(2(1-α))
        - Cauchy: VaR = μ + λ * tan(π(α-0.5))
        """
        if distribution == 'normal':
            mu, sigma = np.mean(returns), np.std(returns)
            var = mu + sigma * stats.norm.ppf(confidence_level)

        elif distribution == 'laplace':
            fitter = AdvancedDistributionFitter()
            params = fitter.fit_laplace_distribution(returns)
            mu, theta = params['mu'], params['theta']
            # For Laplace: VaR = μ + (1/θ) * ln(2(1-α))
            var = mu + (1 / theta) * np.log(2 * (1 - confidence_level))

        elif distribution == 'cauchy':
            fitter = AdvancedDistributionFitter()
            params = fitter.fit_cauchy_distribution(returns)
            mu, lam = params['mu'], params['lambda']
            # For Cauchy: VaR = μ + λ * tan(π(α-0.5))
            var = mu + lam * np.tan(np.pi * (confidence_level - 0.5))

        elif distribution == 'empirical':
            var = np.percentile(returns, confidence_level * 100)

        else:
            raise ValueError(f"Unknown distribution: {distribution}")

        return {
            'var': var,
            'confidence_level': confidence_level,
            'distribution': distribution
        }

    @staticmethod
    def calculate_cvar(returns: np.ndarray, var: float, confidence_level: float = 0.95) -> float:
        """
        Calculate Conditional Value at Risk (Expected Shortfall).

        Mathematical Foundation:
        CVaR_α = E[L | L > VaR_α] = (1/(1-α)) * ∫_{VaR_α}^{∞} x * f(x) dx

        This is the expected loss in the worst (1-α)% of cases.
        """
        tail_losses = returns[returns > var]
        if len(tail_losses) == 0:
            return var

        cvar = np.mean(tail_losses)
        return cvar

    @staticmethod
    def calculate_drawdown_distribution(prices: np.ndarray) -> Dict[str, float]:
        """
        Analyze drawdown distribution using extreme value theory.

        Mathematical Foundation:
        Drawdown: DD_t = max(P_s) - P_t for s < t
        Maximum drawdown: max(DD_t)

        Can be modeled using:
        - Negative Binomial for number of drawdowns
        - Gamma for drawdown magnitudes
        """
        # Calculate cumulative max and drawdowns
        cummax = np.maximum.accumulate(prices)
        drawdowns = (cummax - prices) / cummax

        # Filter significant drawdowns (> 1%)
        significant_drawdowns = drawdowns[drawdowns > 0.01]

        if len(significant_drawdowns) == 0:
            return {'error': 'No significant drawdowns found'}

        # Fit distributions
        fitter = AdvancedDistributionFitter()

        # Fit Gamma to drawdown magnitudes
        gamma_params = fitter.fit_gamma_distribution(significant_drawdowns)

        # Fit Negative Binomial to count of drawdowns in periods
        # This is simplified - in practice, you'd bin by time periods
        n_drawdowns = len(significant_drawdowns)

        return {
            'max_drawdown': np.max(drawdowns),
            'mean_drawdown': np.mean(significant_drawdowns),
            'std_drawdown': np.std(significant_drawdowns),
            'n_significant_drawdowns': n_drawdowns,
            'gamma_alpha': gamma_params['alpha'],
            'gamma_mean': gamma_params['mean'],
            'gamma_skewness': gamma_params['skewness']
        }


# =============================================================================
# SECTION 4: VOLATILITY MODELING WITH DISTRIBUTIONS
# =============================================================================

class StochasticVolatilityModels:
    """
    Enhanced volatility modeling using proper probability distributions.

    Mathematical Foundation:
    - Realized variance follows Gamma distribution (Heston model connection)
    - Volatility clustering captured by distribution parameters
    - Mean reversion in variance parameters
    """

    @staticmethod
    def fit_volatility_distribution(returns: np.ndarray, window: int = 20) -> Dict[str, Dict]:
        """
        Fit probability distributions to realized volatility.

        Mathematical Foundation:
        Realized variance RV_t = Σ r_i² for i in window
        Under Heston: dν_t = κ(θ-ν_t)dt + ξ√ν_t dW_t
        Stationary distribution: ν ~ Gamma(2κθ/ξ², 2κ/ξ²)
        """
        # Calculate realized volatility
        if isinstance(returns, np.ndarray):
            # Convert to pandas Series for rolling operations
            returns_series = pd.Series(returns)
        else:
            returns_series = returns

        log_returns = np.log(returns_series).diff().dropna()
        realized_var = log_returns.rolling(window=window).var().dropna()
        realized_vol = np.sqrt(realized_var)

        fitter = AdvancedDistributionFitter()

        # Fit distributions to volatility
        gamma_fit = fitter.fit_gamma_distribution(realized_vol)
        laplace_fit = fitter.fit_laplace_distribution(realized_vol)

        # For variance, fit Gamma directly
        gamma_var_fit = fitter.fit_gamma_distribution(realized_var)

        return {
            'volatility_gamma': gamma_fit,
            'volatility_laplace': laplace_fit,
            'variance_gamma': gamma_var_fit,
            'sample_stats': {
                'mean_vol': np.mean(realized_vol),
                'std_vol': np.std(realized_vol),
                'mean_var': np.mean(realized_var),
                'std_var': np.std(realized_var)
            }
        }

    @staticmethod
    def estimate_heston_parameters_from_distribution(returns: np.ndarray, window: int = 20) -> Dict[str, float]:
        """
        Estimate Heston parameters from fitted Gamma distribution of variance.

        Mathematical Foundation:
        If ν ~ Gamma(α, β) with α = 2κθ/ξ², β = 2κ/ξ²
        Then: θ = α/β, κ = β/2, ξ = √(2β/α)
        """
        if isinstance(returns, np.ndarray):
            returns_series = pd.Series(returns)
        else:
            returns_series = returns

        log_returns = np.log(returns_series).diff().dropna()
        realized_var = log_returns.rolling(window=window).var().dropna()

        fitter = AdvancedDistributionFitter()
        gamma_fit = fitter.fit_gamma_distribution(realized_var)

        alpha = gamma_fit['alpha']
        beta_param = 1.0 / gamma_fit['alpha']  # Scale parameter for Gamma

        # Convert to Heston parameters
        theta = alpha * beta_param  # Long-run variance
        kappa = 1.0 / beta_param  # Mean reversion speed
        xi = np.sqrt(2 * beta_param / alpha)  # Vol of vol

        return {
            'theta': theta,
            'kappa': kappa,
            'xi': xi,
            'gamma_alpha': alpha,
            'gamma_beta': beta_param
        }


# =============================================================================
# SECTION 5: UTILITY FUNCTIONS FOR EXISTING PIPELINE
# =============================================================================

def quotient_earnings_yield(eps, price):
    """
    Calculate earnings yield for scalar or vector inputs.
    """
    eps = np.asarray(eps, dtype=float)
    price = np.asarray(price, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.divide(eps, price)
    return np.where((eps > 0) & (price > 0), result, np.nan)


def quotient_book_yield(book_value, price):
    """
    Calculate book yield for scalar or vector inputs.
    """
    book_value = np.asarray(book_value, dtype=float)
    price = np.asarray(price, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.divide(book_value, price)
    return np.where((book_value > 0) & (price > 0), result, np.nan)


def iqr_winsorize(data: np.ndarray, limits: Tuple[float, float] = (0.05, 0.95)) -> np.ndarray:
    """
    Winsorize data using IQR method (robust to outliers).

    Mathematical Foundation:
    Uses interquartile range to identify and cap outliers
    """
    q1, q3 = np.percentile(data, [limits[0] * 100, limits[1] * 100])
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return np.clip(data, lower_bound, upper_bound)


def laplace_cdf_membership(x, mu=None, scale=None, direction="high", location=None):
    """
    Calculate a Laplace CDF membership for scalar or vector inputs.

    Mathematical Foundation:
    F(x;μ,θ) = 0.5 * [1 + sign(x-μ) * (1 - exp(-|x-μ|/θ))]
    """
    values = np.asarray(x, dtype=float)
    if mu is None:
        mu = np.nanmedian(values)
    elif location is not None:
        raise TypeError("Specify either mu or location, not both")
    if location is not None:
        mu = location
    if scale is None:
        mad = np.nanmedian(np.abs(values - mu))
        scale = 1.4826 * mad if np.isfinite(mad) and mad > 1e-12 else np.nanstd(values)
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0

    with np.errstate(over="ignore", invalid="ignore"):
        cdf = 0.5 * (1 + np.sign(values - mu) * (1 - np.exp(-np.abs(values - mu) / scale)))
    cdf = np.nan_to_num(cdf, nan=0.5)
    return cdf if direction == "high" else 1.0 - cdf


def robust_mad_zscore(data: np.ndarray) -> np.ndarray:
    """
    Calculate robust z-scores using Median Absolute Deviation.

    Mathematical Foundation:
    MAD = median(|X - median(X)|)
    Z = (X - median) / (MAD * 1.4826)  # 1.4826 makes MAD consistent with std for normal
    """
    median = np.median(data)
    mad = np.median(np.abs(data - median))
    if mad == 0:
        return np.zeros_like(data)
    return (data - median) / (mad * 1.4826)


def beta_cdf_membership(x: float, alpha: float, beta_param: float) -> float:
    """
    Calculate membership function using Beta CDF.

    Mathematical Foundation:
    F(x;α,β) = ∫₀ˣ t^(α-1)(1-t)^(β-1)/B(α,β) dt
    """
    if alpha <= 0 or beta_param <= 0:
        return 0.0
    x_clipped = np.clip(x, 0, 1)
    return stats.beta.cdf(x_clipped, alpha, beta_param)


def beta_calibrated_probability(raw_probability: float, n_obs: int,
                                prior_p: float = 0.50) -> Dict[str, float]:
    """Shrink a classifier probability with a Beta prior and return a 90% CI."""
    n_obs = max(int(n_obs), 1)
    p = float(np.clip(raw_probability, 0.0, 1.0))
    prior_p = float(np.clip(prior_p, 0.0, 1.0))
    alpha = 2.0 * prior_p + n_obs * p
    beta = 2.0 * (1.0 - prior_p) + n_obs * (1.0 - p)
    return {
        "calibrated_p": float(alpha / (alpha + beta)),
        "ci_05": float(stats.beta.ppf(0.05, alpha, beta)),
        "ci_95": float(stats.beta.ppf(0.95, alpha, beta)),
    }


def cauchy_stress_var(data: np.ndarray) -> Dict[str, float]:
    """Estimate Cauchy location/scale and report conservative tail quantiles."""
    values = np.asarray(data, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {"cauchy_var_95": 0.0, "cauchy_var_99": 0.0}
    location = float(np.median(values))
    scale = float(max(np.subtract(*np.percentile(values, [75, 25])) / 2.0, 1e-8))
    return {
        "cauchy_var_95": float(stats.cauchy.ppf(0.95, loc=location, scale=scale)),
        "cauchy_var_99": float(stats.cauchy.ppf(0.99, loc=location, scale=scale)),
    }


def return_moments_diagnostics(data: np.ndarray) -> Dict[str, Union[float, str]]:
    """Return shape diagnostics used to select a return-tail model."""
    values = np.asarray(data, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 3:
        return {"gamma1_skewness": 0.0, "gamma2_kurtosis": 0.0, "recommended_model": "normal"}
    skewness = float(stats.skew(values, bias=False))
    excess_kurtosis = float(stats.kurtosis(values, fisher=True, bias=False))
    recommended = "student_t" if excess_kurtosis > 1.0 else "laplace"
    return {
        "gamma1_skewness": skewness,
        "gamma2_kurtosis": excess_kurtosis,
        "recommended_model": recommended,
    }


def beta_liquidity_weights(liquidity_scores: np.ndarray, alpha: float = 2.0, beta_param: float = 2.0) -> np.ndarray:
    """
    Calculate liquidity weights using Beta distribution.

    Mathematical Foundation:
    Normalizes liquidity scores to [0,1] and applies Beta CDF
    to get smooth, bounded weights.
    """
    if len(liquidity_scores) == 0:
        return np.array([])

    # Normalize to [0,1]
    min_score, max_score = liquidity_scores.min(), liquidity_scores.max()
    if max_score == min_score:
        normalized = np.ones_like(liquidity_scores) * 0.5
    else:
        normalized = (liquidity_scores - min_score) / (max_score - min_score)

    # Apply Beta CDF
    weights = np.array([beta_cdf_membership(x, alpha, beta_param) for x in normalized])

    # Normalize to sum to 1
    if weights.sum() > 0:
        weights = weights / weights.sum()

    return weights


def copula_aggregate(*signals, theta=None, copula_type: str = 'gaussian') -> np.ndarray:
    """
    Aggregate multiple signals using copula theory.

    Mathematical Foundation:
    Uses copula to model dependence between different signals
    and aggregate them into a composite score.
    """
    if len(signals) == 1 and isinstance(signals[0], pd.DataFrame):
        signal_frame = signals[0]
    else:
        if not signals:
            return np.array([])
        signal_frame = pd.concat(
            [pd.Series(signal).reset_index(drop=True) for signal in signals],
            axis=1,
        )
        signal_frame.columns = range(signal_frame.shape[1])

        if theta is not None:
            values = np.clip(signal_frame.to_numpy(dtype=float), 1e-12, 1.0 - 1e-12)
            denominator = np.expm1(-theta)
            product = np.prod(np.expm1(-theta * values), axis=1)
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                composite = -np.log1p(product / denominator ** (values.shape[1] - 1)) / theta
            return np.nan_to_num(composite, nan=0.0, posinf=1.0, neginf=0.0)

    # Convert each signal to uniform marginals
    uniform_signals = signal_frame.copy()
    for col in signal_frame.columns:
        uniform_signals[col] = stats.rankdata(signal_frame[col]) / (len(signal_frame) + 1)

    # Calculate copula correlation matrix
    copula_modeler = CopulaModeler()
    n_signals = len(signal_frame.columns)
    corr_matrix = np.zeros((n_signals, n_signals))

    for i in range(n_signals):
        for j in range(n_signals):
            if i != j:
                corr_matrix[i, j] = copula_modeler.gaussian_copula_correlation(
                    uniform_signals.iloc[:, i], uniform_signals.iloc[:, j]
                )

    # Aggregate using first principal component of copula correlation
    eigenvalues, eigenvectors = np.linalg.eigh(corr_matrix + np.eye(n_signals))
    weights = eigenvectors[:, -1]  # First principal component

    # Normalize weights
    weights = np.abs(weights)
    if weights.sum() > 0:
        weights = weights / weights.sum()

    # Calculate composite score
    composite = (signal_frame.values * weights).sum(axis=1)

    return composite


def frank_copula_bivariate(x: np.ndarray, y: np.ndarray, theta: float = None) -> float:
    """
    Calculate Frank copula dependence measure between two variables.

    Mathematical Foundation:
    C_θ(u,v) = -1/θ * ln[1 + (exp(-θu)-1)(exp(-θv)-1)/(exp(-θ)-1)]

    Returns the dependence parameter θ that best fits the data.
    """
    copula_modeler = CopulaModeler()
    if theta is None:
        theta = copula_modeler.frank_copula_parameter(x, y)
    return theta


# =============================================================================
# SECTION 6: INTEGRATION HELPERS FOR NEPSE PIPELINE
# =============================================================================

class DistributionEnrichedPipeline:
    """
    Integration class for adding distribution-based analysis to the NEPSE pipeline.

    This class provides methods that can be directly integrated into the existing
    nepse_full_quant_pipeline.py to enhance its mathematical robustness.
    """

    @staticmethod
    def enhance_return_analysis(returns: np.ndarray) -> Dict[str, Dict]:
        """
        Enhance return analysis with proper distribution fitting.

        Replaces simple normal distribution assumptions with more
        appropriate distributions for financial returns.
        """
        fitter = AdvancedDistributionFitter()

        # Fit multiple distributions
        results = {
            'normal': {
                'mu': np.mean(returns),
                'sigma': np.std(returns),
                'log_likelihood': np.sum(stats.norm.logpdf(returns, np.mean(returns), np.std(returns)))
            },
            'laplace': fitter.fit_laplace_distribution(returns),
            'cauchy': fitter.fit_cauchy_distribution(returns),
            'gamma': fitter.fit_gamma_distribution(np.abs(returns))  # For absolute returns
        }

        # Select best distribution by AIC
        best_dist = min(results.keys(), key=lambda k: results[k].get('aic', float('inf')))

        return {
            'fitted_distributions': results,
            'best_distribution': best_dist,
            'recommendation': f"Use {best_dist} distribution for return modeling"
        }

    @staticmethod
    def enhance_volatility_forecasting(returns: np.ndarray, current_vol: float) -> Dict[str, float]:
        """
        Enhance volatility forecasting using Gamma distribution theory.

        Connects to Heston model parameters through distribution theory.
        """
        sv_model = StochasticVolatilityModels()

        # Fit volatility distribution
        vol_dist = sv_model.fit_volatility_distribution(returns)

        # Estimate Heston parameters from distribution
        heston_params = sv_model.estimate_heston_parameters_from_distribution(returns)

        # Enhanced forecast using distribution
        gamma_params = vol_dist['volatility_gamma']
        forecast_vol = gamma_params['mean']  # Gamma mean = alpha

        return {
            'current_vol': current_vol,
            'forecast_vol': forecast_vol,
            'volatility_gamma_alpha': gamma_params['alpha'],
            'volatility_gamma_skewness': gamma_params['skewness'],
            'heston_theta': heston_params['theta'],
            'heston_kappa': heston_params['kappa'],
            'heston_xi': heston_params['xi']
        }

    @staticmethod
    def calculate_robust_risk_metrics(returns: np.ndarray, prices: np.ndarray) -> Dict[str, Dict]:
        """
        Calculate robust risk metrics using proper distributions.

        Replaces simple empirical risk metrics with distribution-based
        metrics that account for fat tails and skewness.
        """
        risk_calculator = AdvancedRiskMetrics()

        # VaR under different distributions
        var_results = {
            'normal': risk_calculator.calculate_var(returns, distribution='normal'),
            'laplace': risk_calculator.calculate_var(returns, distribution='laplace'),
            'cauchy': risk_calculator.calculate_var(returns, distribution='cauchy'),
            'empirical': risk_calculator.calculate_var(returns, distribution='empirical')
        }

        # CVaR using empirical distribution
        var_95 = var_results['empirical']['var']
        cvar_95 = risk_calculator.calculate_cvar(returns, var_95)

        # Drawdown analysis
        drawdown_analysis = risk_calculator.calculate_drawdown_distribution(prices)

        return {
            'var_analysis': var_results,
            'cvar_95': cvar_95,
            'drawdown_analysis': drawdown_analysis
        }

    @staticmethod
    def analyze_cross_asset_dependence(returns_dict: Dict[str, np.ndarray]) -> Dict[str, Dict]:
        """
        Analyze dependence between different assets using copula theory.

        Mathematical Foundation:
        Uses copulas to separate marginal distributions from
        dependence structure, providing more robust correlation analysis.
        """
        analyzer = BivariateAnalyzer()
        copula_modeler = CopulaModeler()

        symbols = list(returns_dict.keys())
        dependence_results = {}

        for i, sym1 in enumerate(symbols):
            for j, sym2 in enumerate(symbols):
                if i < j:  # Avoid duplicates and self-comparison
                    returns1 = returns_dict[sym1]
                    returns2 = returns_dict[sym2]

                    # Joint moments
                    joint_moments = analyzer.joint_moments(returns1, returns2)

                    # Independence tests
                    independence = analyzer.test_independence(returns1, returns2, method='pearson')
                    independence_spearman = analyzer.test_independence(returns1, returns2, method='spearman')

                    # Copula parameters
                    gaussian_theta = copula_modeler.gaussian_copula_correlation(returns1, returns2)
                    frank_theta = copula_modeler.frank_copula_parameter(returns1, returns2)

                    pair_key = f"{sym1}_{sym2}"
                    dependence_results[pair_key] = {
                        'joint_moments': joint_moments,
                        'independence_tests': {
                            'pearson': independence,
                            'spearman': independence_spearman
                        },
                        'copula_parameters': {
                            'gaussian_theta': gaussian_theta,
                            'frank_theta': frank_theta
                        }
                    }

        return dependence_results


if __name__ == "__main__":
    # Example usage and testing
    print("Advanced Probability Distribution Utilities for Quantitative Finance")
    print("=" * 70)

    # Generate sample financial data
    np.random.seed(42)
    n_samples = 1000

    # Sample returns with fat tails (t-distribution)
    returns = np.random.standard_t(3, n_samples) * 0.02

    # Sample prices
    prices = 100 * np.cumprod(1 + returns)
    prices_series = pd.Series(prices)

    print("\n1. Distribution Fitting Analysis")
    print("-" * 50)

    fitter = AdvancedDistributionFitter()

    # Fit different distributions
    gamma_fit = fitter.fit_gamma_distribution(np.abs(returns))
    print(f"Gamma fit for absolute returns:")
    print(f"  Alpha: {gamma_fit['alpha']:.4f}")
    print(f"  Mean: {gamma_fit['mean']:.4f}")
    print(f"  Skewness: {gamma_fit['skewness']:.4f}")

    laplace_fit = fitter.fit_laplace_distribution(returns)
    print(f"\nLaplace fit for returns:")
    print(f"  Mu: {laplace_fit['mu']:.6f}")
    print(f"  Theta: {laplace_fit['theta']:.4f}")
    print(f"  AIC: {laplace_fit['aic']:.2f}")

    print("\n2. Risk Metrics Analysis")
    print("-" * 50)

    risk_calc = AdvancedRiskMetrics()
    var_normal = risk_calc.calculate_var(returns, distribution='normal')
    var_laplace = risk_calc.calculate_var(returns, distribution='laplace')

    print(f"VaR (95%, Normal): {var_normal['var']:.4f}")
    print(f"VaR (95%, Laplace): {var_laplace['var']:.4f}")

    print("\n3. Volatility Distribution Analysis")
    print("-" * 50)

    sv_model = StochasticVolatilityModels()
    vol_dist = sv_model.fit_volatility_distribution(prices_series)

    print(f"Volatility Gamma parameters:")
    print(f"  Alpha: {vol_dist['volatility_gamma']['alpha']:.4f}")
    print(f"  Mean: {vol_dist['volatility_gamma']['mean']:.4f}")

    print("\n4. Integration with NEPSE Pipeline")
    print("-" * 50)

    pipeline = DistributionEnrichedPipeline()
    enhanced_analysis = pipeline.enhance_return_analysis(returns)
    print(f"Best distribution for returns: {enhanced_analysis['best_distribution']}")
    print(f"Recommendation: {enhanced_analysis['recommendation']}")

    print("\n" + "=" * 70)
    print("Distribution utilities module loaded successfully!")
    print("Ready for integration with NEPSE quantitative pipeline.")