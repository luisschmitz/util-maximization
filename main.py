from functions import *

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
        print(f"alpha={alpha:.2f}: analytic π*={analytic_pi:.2f}, numeric π*={pi_det:.2f}")
    return results


def main():
    # Parameters
    T = 1.0
    M = 100_000  # Monte Carlo paths
    b, sigma, x0 = 0.1, 0.2, 1.0
    distribution = "constant"
    dist_params = {"const_value": 0.5}

    alphas = [0.5, 1.0, 2.0, 5.0]

    # Run tests with adaptive grids
    results = test_multiple_alphas(
        alphas, T, M, b, sigma, x0,
        distribution, dist_params,
        paths_seed=42
    )

    # Plot π* vs alpha
    alphas_arr = np.array([r["alpha"] for r in results])
    analytic_pis = np.array([r["analytic_pi"] for r in results])
    numeric_pis = np.array([r["numeric_pi"] for r in results])
    plt.figure()
    plt.plot(alphas_arr, analytic_pis, 'o-', label='Analytic π*')
    plt.plot(alphas_arr, numeric_pis, 'x--', label='Numeric π*')
    plt.xlabel('alpha')
    plt.ylabel('pi*')
    plt.title('Optimal π* vs Risk Aversion α')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot utility curves for each alpha with its adaptive grid
    for r in results:
        plot_utilities(
            r["pi_grid"], r["eu_vals"],
            title=f'Expected Utility vs π (α={r["alpha"]:.1f})',
            pi_opt=r["analytic_pi"]
        )
        
if __name__ == "__main__":
    main()
