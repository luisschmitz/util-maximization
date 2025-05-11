import numpy as np
import matplotlib.pyplot as plt


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

    # 1) Hedgeable Gaussian liability: F depends on W_T
    if distribution == "hedgable_gaussian":
        mu_F = dist_params.get("mu_F", 0.0)
        sigma_F = dist_params.get("sigma_F", 0.0)
        # Scaling factor so that Var[kappa * W_T] = sigma_F^2
        kappa = sigma_F / np.sqrt(T)
        # F = mu_F + kappa * W_T
        F_vec = mu_F + kappa * W_T

    # 2) Independent normal liability: no exposure to W_T, hedging term drops out
    elif distribution == "normal":
        # For purely independent F, the optimal pi is analytic and independent of F
        F_vec = np.zeros(M)

    # 3) Constant liability
    elif distribution == "constant":
        F_vec = generate_F(M, distribution, **dist_params, seed=seed)

    else:
        raise ValueError(f"Unsupported distribution '{distribution}' in find_optimal_pi")

    eu_vals = np.array([
        expected_utility(pi, W_T, F_vec, x0, b, sigma, alpha, T)
        for pi in pi_grid
    ])

    # Pick the pi that maximizes expected utility
    best_pi = pi_grid[eu_vals.argmax()]
    return best_pi, eu_vals


def plot_utilities(pi_grid: np.ndarray,
                   eu_vals: np.ndarray,
                   title: str,
                   pi_opt: float = None) -> None:
    """Plot expected utility vs π and mark the optimal π if provided."""
    plt.plot(pi_grid, eu_vals, label=title)
    if pi_opt is not None:
        plt.axvline(pi_opt, color='r', linestyle='--', label=f"Optimal π ≈ {pi_opt:.2f}")
    plt.title(title)
    plt.xlabel("π")
    plt.ylabel("Expected Utility")
    plt.legend()
    plt.grid(True)
    plt.show()