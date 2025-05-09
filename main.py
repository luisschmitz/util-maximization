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

        # sample liabilities: constant or zero for normal
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
    # simulate fresh W_T for utility curves
    W_T_common = simulate_WT(T, M, seed=42)

    plt.figure(figsize=(8, 5))
    for r in results:
        alpha = r["alpha"]
        analytic = r["analytic_pi"]
        numeric = r["numeric_pi"]
        # F_vec: constant vs zero for normal
        F_vec = np.zeros_like(W_T_common)
        if distribution == "constant":
            F_vec = np.full_like(W_T_common, dist_params["const_value"])
        # compute utility
        eu_vals = [expected_utility(pi, W_T_common, F_vec, x0, b, sigma, alpha, T)
                   for pi in pi_grid_common]
        plt.plot(pi_grid_common, eu_vals,
                 label=f"α={alpha} (ana={analytic:.2f}, num={numeric:.2f})")
        # analytic pi
        plt.axvline(analytic, linestyle='--', color='k')
        # numeric pi marker
        idx = np.argmin(np.abs(pi_grid_common - numeric))
        plt.plot(numeric, eu_vals[idx], 'o', color='k')

    plt.xlabel('π')
    plt.ylabel('Expected Utility')
    plt.title(f'Expected Utility vs π for Different α — {distribution} F')
    plt.legend(title='α (ana, num)')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def main():
    # Base parameters
    T = 1.0
    M = 100_000
    b, sigma, x0 = 0.1, 0.2, 1.0
    alphas = [0.5, 1.0, 2.0, 5.0]

    # Define scenarios for constant and normal liabilities
    scenarios = [
        ("constant", {"const_value": 0.5}),
        ("normal",   {"normal_mu": 0.5, "normal_sigma": 1.0})
    ]

    for distribution, dist_params in scenarios:
        print(f"\n--- Scenario: {distribution} F ---")
        results = test_multiple_alphas(
            alphas, T, M, b, sigma, x0,
            distribution, dist_params,
            paths_seed=42
        )
        plot_combined_alpha(results, T, M, b, sigma, x0,
                            distribution, dist_params)

if __name__ == "__main__":
    main()
