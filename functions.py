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

def expected_power_utility(
    x0: float,
    b: float,
    sigma: float,
    gamma: float,
    T: float,
    distribution: str,
    dist_params: dict,
    r: float = 0.0,
    n_steps: int = 200,
    M: int = 500_000,
    seed: int = None
) -> float:
    """
    Monte-Carlo E[U(X_T - F)] for CRRA U(x)=x^γ/γ via feedback π_t = H_t/(1−γ)*(b/σ²),
    falling back to log-utility if γ≈1.
    """
    # 1) γ=1: use log-utility closed-form
    if abs(gamma - 1.0) < 1e-8:
        pi_star = b / sigma**2
        rng = np.random.default_rng(seed)
        W_T = rng.normal(0.0, np.sqrt(T), size=M)
        # build F_vec
        if distribution == 'hedgable_gaussian':
            muF = dist_params['mu_F']
            sigmaF = dist_params['sigma_F']
            kappa = sigmaF / np.sqrt(T)
            F_vec = muF + kappa * W_T
        elif distribution == 'normal':
            F_vec = np.zeros(M)
        else:
            F_vec = np.full(M, dist_params['const_value'])
        return expected_log_utility(pi_star, W_T, F_vec, x0, b, sigma, T)

    # 2) Otherwise CRRA γ≠1
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    theta = b / sigma

    # simulate dW increments and cumulative W_path
    dW = rng.normal(0.0, np.sqrt(dt), size=(M, n_steps))
    W_path = np.cumsum(dW, axis=1)  # shape (M, n_steps)
    W_T = W_path[:, -1]

    # set liability F and Y-drift
    if distribution == 'hedgable_gaussian':
        muF = dist_params['mu_F']
        sigmaF = dist_params['sigma_F']
        kappa = sigmaF / np.sqrt(T)
        F = muF + kappa * W_T
        drift_Y = kappa*theta + (gamma/(2*(1-gamma))) * theta**2
    elif distribution == 'normal':
        F = np.zeros(M)
        drift_Y = (gamma/(2*(1-gamma))) * theta**2
    else:  # constant
        constF = dist_params['const_value']
        F = np.full(M, constF)
        drift_Y = (gamma/(2*(1-gamma))) * theta**2

    # initialize X, Y
    X = np.full(M, x0)
    if distribution == 'hedgable_gaussian':
        Y = np.full(M, muF - drift_Y*T)  # at t=0, W_path=0
    else:
        barF = float(np.mean(F))
        Y = np.full(M, barF - drift_Y*T)

    # time-stepping (reuse dW for both X and Y)
    for i in range(n_steps):
        t = i * dt
        dW_i = dW[:, i]

        # update Y_t
        if distribution == 'hedgable_gaussian':
            Wt = W_path[:, i]
            Y = muF + kappa*Wt - drift_Y*(T - t)

        # feedback π
        H = X - Y
        pi_t = H/(1-gamma) * (b / sigma**2)

        # Euler–Maruyama for X
        X = X + (r*X + pi_t*b) * dt + pi_t*sigma * dW_i

    # compute U(X_T - F)
    surplus = X - F
    is_int_gamma = abs(gamma - round(gamma)) < 1e-8

    if is_int_gamma:
        U = surplus**gamma / gamma
    else:
        U = np.zeros_like(surplus)
        mask = surplus > 0
        U[mask] = surplus[mask]**gamma / gamma

    return U.mean()

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