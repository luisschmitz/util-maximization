import numpy as np

def simulate_WT(T: float, M: int, seed: int = None) -> np.ndarray:
    """Draw M samples of W_T ~ N(0, T)."""
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, np.sqrt(T), size=M)

def generate_F(M: int, distribution: str, const_value: float = 0.0, normal_mu: float = 0.0, normal_sigma: float = 1.0, seed: int = None) -> np.ndarray:
    """
    Generate M liabilities according to `distribution`.
    Currently supported options:
    - "constant": array of const_value.
    - "normal": N(normal_mu, normal_sigma^2) samples.
    """
    rng = np.random.default_rng(seed)
    if distribution == "constant":
        return np.full(M, const_value)
    elif distribution == "normal":
        return rng.normal(loc=normal_mu, scale=normal_sigma, size=M)
    else:
        raise ValueError(f"Unsupported distribution '{distribution}'")
    

def expected_utility(pi: float, W_T: np.ndarray, F_vec: np.ndarray, x0: float, b: float, sigma: float, alpha: float) -> float:
    """Compute E[−exp(−α·(X_T − F))] under strategy π."""
    X_T = x0 + pi * (b * 1.0 + sigma * W_T)
    U = -np.exp(-alpha * (X_T - F_vec))
    return U.mean()

def find_optimal_pi(pi_grid: np.ndarray, W_T: np.ndarray, distribution: str, dist_params: dict, x0: float, b: float, sigma: float, alpha: float, seed: int = None) -> tuple[float, np.ndarray]:
    """
    Sweep π over pi_grid; for each:
      1) draw F_vec = generate_F(M, distribution, **dist_params, seed)
      2) compute its expected utility.
    Return (best_pi, all_EU_values).
    """
    M = W_T.shape[0]
    eu_vals = []
    for pi in pi_grid:
        F_vec = generate_F(M, distribution, **dist_params, seed=seed)
        eu_vals.append(expected_utility(pi, W_T, F_vec, x0, b, sigma, alpha))
    eu_vals = np.array(eu_vals)
    best_pi = pi_grid[eu_vals.argmax()]
    return best_pi, eu_vals