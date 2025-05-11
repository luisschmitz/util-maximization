import numpy as np

def simulate_WT(T: float, M: int, seed: int = None) -> np.ndarray:
    """Draw M samples of W_T ~ N(0, T)."""
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, np.sqrt(T), size=M)


def generate_F(M: int,
               distribution: str,
               const_value: float = 0.0,
               normal_mu: float = 0.0,
               normal_sigma: float = 1.0,
               seed: int = None) -> np.ndarray:
    """
    Generate M liabilities according to `distribution`.
    Supported:
      - "constant": array of const_value.
      - "normal": iid N(normal_mu, normal_sigma^2) samples.
    Note: For the hedgeable Gaussian-liability case, we handle it separately
    in `find_optimal_pi`, since it must depend on the same W_T samples.
    """
    rng = np.random.default_rng(seed)
    if distribution == "constant":
        return np.full(M, const_value)
    elif distribution == "normal":
        return rng.normal(loc=normal_mu, scale=normal_sigma, size=M)
    else:
        raise ValueError(f"Unsupported distribution '{distribution}'")


def expected_utility(pi: float,
                     W_T: np.ndarray,
                     F_vec: np.ndarray,
                     x0: float,
                     b: float,
                     sigma: float,
                     alpha: float,
                     T: float) -> float:
    """Compute E[-exp(-alpha*(X_T - F))] under strategy pi."""
    # Wealth at time T: X_T = x0 + pi * (b*T + sigma*W_T)
    X_T = x0 + pi * (b * T + sigma * W_T)
    U = -np.exp(-alpha * (X_T - F_vec))
    return U.mean()


def expected_power_utility(pi,     # π ignored, since we use the analytic π*
                           W_T,    # W_T used to infer kappa if needed
                           F_vec,  # F_vec has liability samples
                           x0: float,
                           b: float,
                           sigma: float,
                           gamma: float,
                           T: float) -> float:
    """
    Analytic value function for CRRA (power) utility with bounded liability F.

    Returns V(x0) = (x0 - Y0)^gamma / gamma, where Y0 is the BSDE certainty equivalent:
      - constant F:      F̄ - (γ/[2(1-γ)]) θ^2 T
      - hedgeable Gaussian F=μ_F+κW_T:
                          μ_F - κθ T - (γ/[2(1-γ)]) θ^2 T
    """
    if abs(gamma - 1.0) < 1e-8:
      # avoid division by zero; use log-utility closed-form
      return expected_log_utility(pi, W_T, F_vec, x0, b, sigma, T)
    theta = b / sigma

    # average liability
    try:
      barF = float(np.mean(F_vec))
    except:
      barF = 0.0

    # determine if hedgeable Gaussian (nonzero variance in F_vec)
    if np.std(F_vec) > 1e-8:
      mu_F = np.mean(F_vec)
      # infer kappa via covariance with W_T
      cov = np.cov(F_vec, W_T, bias=True)[0,1]
      varW = np.var(W_T)
      kappa = cov / varW if varW>0 else 0.0
      # Y0 from Gaussian liability
      Y0 = mu_F - kappa * theta * T - (gamma/(2*(1-gamma))) * theta**2 * T
    else:
      # Y0 for constant liability
      Y0 = barF - (gamma/(2*(1-gamma))) * theta**2 * T

    H0 = x0 - Y0
    return H0**gamma / gamma

def expected_log_utility(pi: float,
                         W_T: np.ndarray,
                         F_vec: np.ndarray,
                         x0: float,
                         b: float,
                         sigma: float,
                         T: float) -> float:
    """
    Compute E[log(X_T - F)] under a multiplicative GBM model:
      dX_t = π X_t (b dt + σ dW_t),
    so
      X_T = x0 * exp((π b - 0.5 π^2 σ^2) T + π σ W_T).

    Utility is log(X_T - F_vec), with a large negative penalty if X_T <= F.
    """
    # simulate terminal wealth
    X_T = x0 * np.exp((pi * b - 0.5 * pi**2 * sigma**2) * T + pi * sigma * W_T)

    # surplus above the liability
    S = X_T - F_vec

    # compute log‐utility, penalize non‐positive surplus
    U = np.empty_like(S)
    mask = S > 0
    U[mask] = np.log(S[mask])
    U[~mask] = 0  
    return U.mean()


def find_optimal_pi(pi_grid: np.ndarray,
                    W_T: np.ndarray,
                    distribution: str,
                    dist_params: dict,
                    x0: float,
                    b: float,
                    sigma: float,
                    alpha: float,
                    T: float,
                    seed: int = None) -> tuple[float, np.ndarray]:
    """
    Sweep pi over pi_grid to find the numeric optimum of expected utility.

    For exponential utility with various liability types:
      - "constant": uses generate_F.
      - "normal": treats F as independent normal (no hedging term).
      - "hedgable_gaussian": F = mu_F + kappa * W_T, with kappa = sigma_F / sqrt(T).
    Returns (best_pi, all_EU_values).
    """
    M = W_T.shape[0]

    # Hedgeable Gaussian liability: F depends on W_T
    if distribution == "hedgable_gaussian":
        mu_F = dist_params.get("mu_F", 0.0)
        sigma_F = dist_params.get("sigma_F", 0.0)
        kappa = sigma_F / np.sqrt(T)
        F_vec = mu_F + kappa * W_T

    elif distribution == "normal":
        F_vec = np.zeros(M)
    elif distribution == "constant":
        F_vec = generate_F(M, distribution, **dist_params, seed=seed)
    else:
        raise ValueError(f"Unsupported distribution '{distribution}' in find_optimal_pi")
    
    eu_vals = np.array([
        expected_utility(pi, W_T, F_vec, x0, b, sigma, alpha, T)
        for pi in pi_grid
    ])

    best_pi = pi_grid[eu_vals.argmax()]
    return best_pi, eu_vals