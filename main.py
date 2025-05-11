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
    and numeric pi* annotations, plus analytic closed-form utility.
    """
    pi_min = min(r["pi_grid"][0] for r in results)
    pi_max = max(r["pi_grid"][-1] for r in results)
    pi_grid_common = np.linspace(pi_min, pi_max, 400)
    W_T_common = simulate_WT(T, M, seed=42)

    plt.figure(figsize=(8, 5))
    for r in results:
        alpha = r["alpha"]
        analytic_pi = r["analytic_pi"]
        numeric_pi = r["numeric_pi"]
        if distribution == "constant":
            F_vec = np.full_like(W_T_common, dist_params["const_value"])
            Y0 = dist_params["const_value"] - T * (b/sigma)**2 / (2 * alpha)
        else:
            F_vec = np.zeros_like(W_T_common)
            mu = dist_params.get("normal_mu", 0)
            Y0 = mu - T * (b/sigma)**2 / (2 * alpha)
        # Monte-Carlo utility
        eu_vals = [expected_utility(pi, W_T_common, F_vec, x0, b, sigma, alpha, T)
                   for pi in pi_grid_common]
        plt.plot(pi_grid_common, eu_vals,
                 label=f"MC α={alpha} (num={numeric_pi:.2f})")
        plt.axvline(analytic_pi, linestyle='--', color='k')
        idx = np.argmin(np.abs(pi_grid_common - numeric_pi))
        plt.plot(numeric_pi, eu_vals[idx], 'o', color='k')
        # analytic closed-form utility curve
        U_ana = -np.exp(-alpha * (x0 - Y0 + pi_grid_common * b * T))
        plt.plot(pi_grid_common, U_ana, linestyle=':', label=f"Ana α={alpha}")

    plt.xlabel('π')
    plt.ylabel('Expected Utility')
    plt.title(f'Expected vs Analytic Utility vs π — {distribution.capitalize()} Liability')
    plt.legend(fontsize='small', ncol=2)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def simulate_utility_paths(alpha, pi_star, T, N, Msim, b, sigma, x0, F):
    """
    Simulate Msim sample paths of U_t under rebalanced strategy π_star or time-varying π.
    Returns U array shape (Msim, N+1).
    """
    dt = T / N
    rng = np.random.default_rng(123)
    dW = rng.normal(0, np.sqrt(dt), size=(Msim, N))
    X = np.zeros((Msim, N+1))
    X[:, 0] = x0
    for k in range(N):
        X[:, k+1] = X[:, k] + pi_star[k] * (b[k] * dt + sigma[k] * dW[:, k])
    F_mat = F.reshape(Msim, 1)
    U = -np.exp(-alpha * (X - F_mat))
    return U


def plot_time_varying_strategy(alphas, T, N, b_func, sigma_func, x0):
    """
    Plot dynamic optimal π*(t) for time-varying b(t), σ(t) and simulate utility paths.
    """
    tgrid = np.linspace(0, T, N+1)
    b_t = b_func(tgrid)
    sigma_t = sigma_func(tgrid)

    # Plot π*(t)
    plt.figure(figsize=(8, 4))
    for alpha in alphas:
        pi_t = b_t / (alpha * sigma_t**2)
        plt.plot(tgrid, pi_t, label=f"α={alpha}")
    plt.xlabel('t')
    plt.ylabel('π*(t)')
    plt.title('Time-varying Optimal π*(t)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Simulate utility paths under dynamic π*(t)
    Msim = 2000
    dt = T / N
    tgrid = np.linspace(0, T, N+1)
    for alpha in alphas:
        pi_t = b_t / (alpha * sigma_t**2)
        # constant liability for illustration
        F = np.full(Msim, 0.5)
        U = simulate_utility_paths(alpha, pi_t, T, N, Msim, b_t, sigma_t, x0, F)
        # plot sample paths
        plt.figure(figsize=(8, 4))
        sel = np.random.default_rng(42).choice(Msim, size=5, replace=False)
        for i in sel:
            plt.plot(tgrid, U[i], alpha=0.6)
        plt.xlabel('t')
        plt.ylabel('U_t')
        plt.title(f'Sample U_t Paths (α={alpha}) under Time-Varying b(t),σ(t)')
        plt.grid(True)
        plt.tight_layout()
        plt.show()


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

    # Static plots
    for distribution, dist_params in scenarios:
        print(f"\n--- Scenario: {distribution.capitalize()} Liability ---")
        results = test_multiple_alphas(
            alphas, T, M, b, sigma, x0,
            distribution, dist_params,
            paths_seed=42
        )
        plot_combined_alpha(results, T, M, b, sigma, x0,
                            distribution, dist_params)

    # Sample-path utility plots
    N_steps, Msim = 200, 5000
    tgrid = np.linspace(0, T, N_steps+1)
    for distribution, dist_params in scenarios:
        print(f"\n--- Sample Utility Paths under {distribution.capitalize()} Liability ---")
        if distribution == "constant":
            F_vec = np.full(Msim, dist_params["const_value"])
        else:
            F_vec = generate_F(Msim, distribution, **dist_params, seed=123)
        plt.figure(figsize=(8, 5))
        for alpha in alphas:
            pi_star = b/(alpha*sigma**2)
            U = simulate_utility_paths(alpha, pi_star * np.ones(N_steps+1), T, N_steps, Msim, b*np.ones(N_steps+1), sigma*np.ones(N_steps+1), x0, F_vec)
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

    # Time-varying b(t), sigma(t) demo
    print("\n--- Time-Varying b(t), σ(t) Strategy ---")
    N = 200
    b_func = lambda t: 0.1 + 0.05 * np.sin(2 * np.pi * t / T)
    sigma_func = lambda t: 0.2 + 0.1 * np.cos(2 * np.pi * t / T)
    plot_time_varying_strategy(alphas, T, N, b_func, sigma_func, x0)

if __name__ == "__main__":
    main()
