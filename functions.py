import numpy as np
import matplotlib.pyplot as plt

def simulate_WT(T: float, M: int, seed: int = None) -> np.ndarray:
    """Draw M samples of W_T ~ N(0, T)."""
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, np.sqrt(T), size=M)


def generate_F(M: int, distribution: str, const_value: float = 0.0,
               normal_mu: float = 0.0, normal_sigma: float = 1.0,
               seed: int = None) -> np.ndarray:
    """
    Generate M liabilities according to `distribution`.
    Supported:
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


def expected_utility(pi: float,
                     W_T: np.ndarray,
                     F_vec: np.ndarray,
                     x0: float,
                     b: float,
                     sigma: float,
                     alpha: float,
                     T: float) -> float:
    """Compute E[-exp(-alpha*(X_T - F))] under strategy pi."""
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
    Sweep pi over pi_grid; for each:
      - for constant F: draw one sample vector F_vec
      - for normal F: skip sampling (use F_vec = zeros) since pi* independent of F
    Return (best_pi, all_EU_values).
    """
    M = W_T.shape[0]
    if distribution == "normal":
        # optimal pi is independent of F, so use zero liability sample
        F_vec = np.zeros(M)
    else:
        # draw liabilities once for constant or other distributions
        F_vec = generate_F(M, distribution, **dist_params, seed=seed)

    eu_vals = [expected_utility(pi, W_T, F_vec, x0, b, sigma, alpha, T)
               for pi in pi_grid]
    eu_vals = np.array(eu_vals)
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