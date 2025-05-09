from functions import *
import numpy as np
import matplotlib.pyplot as plt

def test_multiple_alphas(alphas, T, M, b, sigma, x0, distribution, dist_params, paths_seed, grid_points=300):
    """
    For each alpha in alphas, compute analytic and numeric optimal pi,
    using an adaptive search grid around the analytic value.
    Returns a list of result dicts.
    """
    W_T = simulate_WT(T, M, seed=paths_seed)
    results = []
    for alpha in alphas:
        analytic_pi = b / (alpha * sigma**2)
        # adaptive grid around analytic optimum
        window = max(1.0, 0.5 * analytic_pi)
        lower = max(0.0, analytic_pi - window)
        upper = analytic_pi + window
        pi_grid_alpha = np.linspace(lower, upper, grid_points)

        pi_det, eu_vals = find_optimal_pi(
            pi_grid_alpha, W_T,
            distribution, dist_params,
            x0, b, sigma, alpha, T,
            seed=paths_seed
        )
        results.append({
            "alpha": alpha,
            "analytic_pi": analytic_pi,
            "numeric_pi": pi_det,
            "pi_grid": pi_grid_alpha,
            "eu_vals": eu_vals
        })
        print(f"[{distribution}] alpha={alpha:.2f}: analytic π*={analytic_pi:.2f}, numeric π*={pi_det:.2f}")
    return results


def plot_combined_alpha(results, T, M, b, sigma, x0, distribution, dist_params):
    """
    Plot overlaid expected utility curves for different alphas, with analytic
    and numeric pi* annotations, for one distribution scenario.
    """
    # Common pi grid spanning all adaptive grids
    pi_min = min(r["pi_grid"][0] for r in results)
    pi_max = max(r["pi_grid"][-1] for r in results)
    pi_grid_common = np.linspace(pi_min, pi_max, 400)
    W_T_common = simulate_WT(T, M, seed=42)

    plt.figure(figsize=(8, 5))
    for r in results:
        alpha = r["alpha"]
        analytic = r["analytic_pi"]
        numeric = r["numeric_pi"]
        # prepare liability vector
        if distribution == "constant":
            F_vec = np.full_like(W_T_common, dist_params["const_value"])
        else:
            F_vec = np.zeros_like(W_T_common)
        # compute utility curve
        eu_vals = [expected_utility(pi, W_T_common, F_vec, x0, b, sigma, alpha, T)
                   for pi in pi_grid_common]
        plt.plot(pi_grid_common, eu_vals,
                 label=f"α={alpha} (ana={analytic:.2f}, num={numeric:.2f})")
        plt.axvline(analytic, linestyle='--', color='k')
        idx = np.argmin(np.abs(pi_grid_common - numeric))
        plt.plot(numeric, eu_vals[idx], 'o', color='k')

    plt.xlabel('π')
    plt.ylabel('Expected Utility')
    plt.title(f'Expected Utility vs π — {distribution.capitalize()} Liability')
    plt.legend(title='α (ana, num)')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def simulate_utility_paths(alpha, pi_star, T, N, Msim, b, sigma, x0, F):
    """
    Simulate Msim sample paths of U_t under constant rebalanced strategy π_star.
    Returns U array shape (Msim, N+1).
    """
    dt = T / N
    rng = np.random.default_rng(123)
    dW = rng.normal(0, np.sqrt(dt), size=(Msim, N))
    X = np.zeros((Msim, N+1))
    X[:, 0] = x0
    for k in range(N):
        X[:, k+1] = X[:, k] + pi_star * (b * dt + sigma * dW[:, k])
    # ensure F broadcasts along time dimension
    F_mat = F.reshape(Msim, 1)
    U = -np.exp(-alpha * (X - F_mat))
    return U


def main():
    # Base parameters
    T = 1.0
    M = 100_000
    b, sigma, x0 = 0.1, 0.2, 1.0
    alphas = [0.5, 1.0, 2.0, 5.0]

    # Scenarios
    scenarios = [
        ("constant", {"const_value": 0.5}),
        ("normal",   {"normal_mu": 0.5, "normal_sigma": 1.0})
    ]

    # Static plots for each scenario
    for distribution, dist_params in scenarios:
        print(f"\n--- Scenario: {distribution.capitalize()} Liability ---")
        results = test_multiple_alphas(
            alphas, T, M, b, sigma, x0,
            distribution, dist_params,
            paths_seed=42
        )
        plot_combined_alpha(results, T, M, b, sigma, x0,
                            distribution, dist_params)

    # Sample-path utility plots for each scenario
    N_steps, Msim = 200, 5000
    tgrid = np.linspace(0, T, N_steps+1)
    for distribution, dist_params in scenarios:
        print(f"\n--- Sample Utility Paths under {distribution.capitalize()} Liability ---")
        # Prepare F vector for simulation
        if distribution == "constant":
            F_vec = np.full(Msim, dist_params["const_value"])
        else:
            F_vec = generate_F(Msim, distribution, **dist_params, seed=123)

        plt.figure(figsize=(8, 5))
        for alpha in alphas:
            pi_star = b/(alpha*sigma**2)
            U = simulate_utility_paths(alpha, pi_star, T, N_steps, Msim, b, sigma, x0, F_vec)
            sel = np.random.default_rng(42).choice(Msim, size=5, replace=False)
            for i in sel:
                plt.plot(tgrid, U[i], alpha=0.6)
        for alpha in alphas:
            ps = b/(alpha*sigma**2)
            plt.plot([], [], label=f"α={alpha}, π*={ps:.2f}")
        plt.xlabel('t')
        plt.ylabel('Utility U_t')
        plt.title(f'Sample Paths of Utility Process under {distribution.capitalize()} Liability')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()
