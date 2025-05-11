import numpy as np
import matplotlib.pyplot as plt
from functions import simulate_WT, expected_utility, find_optimal_pi, generate_F

# --- Scenarios including hedgeable Gaussian liability ---
scenarios = [
    ("constant",           {"const_value": 0.5}),
    ("normal",             {"normal_mu":   0.5, "normal_sigma": 1.0}),
    ("hedgable_gaussian",  {"mu_F":        0.5,  "sigma_F":     1.0}),
]

# --- Monte-Carlo vs Analytic π* search ---
def test_multiple_alphas(alphas, T, M, b, sigma, x0,
                         distribution, dist_params, paths_seed,
                         grid_points=300):
    W_T = simulate_WT(T, M, seed=paths_seed)
    results = []
    for alpha in alphas:
        # analytic π* depends on the liability type
        if distribution == "hedgable_gaussian":
            kappa = dist_params["sigma_F"] / np.sqrt(T)
            analytic_pi = kappa/sigma + b/(alpha * sigma**2)
        else:
            analytic_pi = b / (alpha * sigma**2)

        # adaptive search grid around analytic π*
        window = max(1.0, 0.5 * analytic_pi)
        lower  = max(0.0, analytic_pi - window)
        upper  = analytic_pi + window
        pi_grid = np.linspace(lower, upper, grid_points)

        # numeric search
        pi_num, eu_vals = find_optimal_pi(
            pi_grid, W_T,
            distribution, dist_params,
            x0, b, sigma, alpha, T,
            seed=paths_seed
        )
        print(f"[{distribution}] α={alpha:.2f}: analytic π*={analytic_pi:.2f}, numeric π*={pi_num:.2f}")

        results.append({
            "alpha":       alpha,
            "analytic_pi": analytic_pi,
            "numeric_pi":  pi_num,
            "pi_grid":     pi_grid,
            "eu_vals":     eu_vals
        })
    return results

# --- Plot Monte-Carlo and analytic utility curves ---
def plot_combined_alpha(results, T, M, b, sigma, x0,
                        distribution, dist_params):
    pi_min = min(r["pi_grid"][0]  for r in results)
    pi_max = max(r["pi_grid"][-1] for r in results)
    pi_grid = np.linspace(pi_min, pi_max, 400)
    W_T = simulate_WT(T, M, seed=42)

    plt.figure(figsize=(8,5))
    for r in results:
        alpha      = r["alpha"]
        analytic_pi= r["analytic_pi"]
        numeric_pi = r["numeric_pi"]

        # construct liability vector and analytic Y0
        if distribution == "constant":
            F_vec = np.full_like(W_T, dist_params["const_value"])
            Y0    = dist_params["const_value"] - T*(b/sigma)**2/(2*alpha)
            U_ana = -np.exp(-alpha*(x0 - Y0 + pi_grid*b*T))

        elif distribution == "normal":
            F_vec = np.zeros_like(W_T)
            mu     = dist_params["normal_mu"]
            Y0     = mu - T*(b/sigma)**2/(2*alpha)
            U_ana  = -np.exp(-alpha*(x0 - Y0 + pi_grid*b*T))

        else:  # hedgable Gaussian
            mu_F    = dist_params["mu_F"]
            sigma_F = dist_params["sigma_F"]
            kappa   = sigma_F / np.sqrt(T)
            theta   = b/sigma
            # analytic Y0 = μ_F - T*(κθ + θ^2/(2α))
            Y0      = mu_F - (kappa*theta + theta**2/(2*alpha))*T
            c       = pi_grid*sigma - kappa
            U_ana   = -np.exp(
                -alpha*(x0 - Y0 + pi_grid*b*T)
                + 0.5*alpha**2 * (c**2) * T
            )
            F_vec   = mu_F + kappa * W_T

        # Monte-Carlo utility curve
        eu_vals = [
            expected_utility(pi, W_T, F_vec, x0, b, sigma, alpha, T)
            for pi in pi_grid
        ]
        plt.plot(pi_grid, eu_vals,
                 label=f"MC α={alpha:.2f} (num={numeric_pi:.2f})")

        # mark analytic & numeric optima
        plt.axvline(analytic_pi, linestyle='--', color='k')
        idx = np.argmin(np.abs(pi_grid - numeric_pi))
        plt.plot(numeric_pi, eu_vals[idx], 'o', color='k')

        # analytic closed-form curve
        plt.plot(pi_grid, U_ana, linestyle=':', label=f"Ana α={alpha:.2f}")

    plt.xlabel('π')
    plt.ylabel('Expected Utility')
    plt.title(f'Exponential Utility — {distribution.replace("_"," ").title()}')
    plt.legend(fontsize='small', ncol=2)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# --- Sample-path utility plots including hedgable Gaussian liability ---
def sample_path_plots(alphas, T, Msim, b, sigma, x0):
    N_steps = 200
    for distribution, dist_params in scenarios:
        print(f"\n--- Sample Utility Paths under {distribution} Liability ---")
        if distribution == "constant":
            F_vec = np.full(Msim, dist_params["const_value"])
        elif distribution == "normal":
            F_vec = generate_F(Msim, distribution, **dist_params, seed=123)
        elif distribution == "hedgable_gaussian":
            mu_F    = dist_params["mu_F"]
            sigma_F = dist_params["sigma_F"]
            kappa   = sigma_F / np.sqrt(T)
            W_Tp    = simulate_WT(T, Msim, seed=123)
            F_vec   = mu_F + kappa * W_Tp
        else:
            raise ValueError(f"Unsupported distribution '{distribution}' in sample paths")

        tgrid = np.linspace(0, T, N_steps+1)
        plt.figure(figsize=(8,5))
        for alpha in alphas:
            pi_star = b/(alpha*sigma**2)
            U = simulate_utility_paths(alpha,
                                       pi_star * np.ones(N_steps+1),
                                       T, N_steps, Msim,
                                       b*np.ones(N_steps+1),
                                       sigma*np.ones(N_steps+1),
                                       x0, F_vec)
            sel = np.random.default_rng(42).choice(Msim, size=5, replace=False)
            for i in sel:
                plt.plot(tgrid, U[i], alpha=0.6)
        for alpha in alphas:
            ps = b/(alpha*sigma**2)
            plt.plot([], [], label=f"α={alpha}, π*={ps:.2f}")
        plt.xlabel('t')
        plt.ylabel('Utility U_t')
        plt.title(f'Sample Paths under {distribution.capitalize()} Liability')
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.show()

# --- Time-varying strategy demo (unchanged) ------------------------------
def simulate_utility_paths(alpha, pi_star, T, N, Msim, b, sigma, x0, F):
    dt = T/N
    rng= np.random.default_rng(123)
    dW = rng.normal(0, np.sqrt(dt), size=(Msim, N))
    X  = np.zeros((Msim, N+1)); X[:,0]=x0
    for k in range(N):
        X[:,k+1] = X[:,k] + pi_star[k]*(b[k]*dt + sigma[k]*dW[:,k])
    U = -np.exp(-alpha*(X - F.reshape(Msim,1)))
    return U


def plot_time_varying_strategy(alphas, T, N, b_func, sigma_func, x0):
    t = np.linspace(0, T, N+1)
    b_t   = b_func(t)
    sigma_t = sigma_func(t)
    plt.figure(figsize=(8,4))
    for alpha in alphas:
        plt.plot(t, b_t/(alpha*sigma_t**2), label=f"α={alpha}")
    plt.xlabel('t'); plt.ylabel('π*(t)')
    plt.title('Time-varying π*(t)'); plt.legend(); plt.grid(); plt.tight_layout(); plt.show()

    Msim=2000
    for alpha in alphas:
        pi_t = b_t/(alpha*sigma_t**2)
        F    = np.full(Msim, 0.5)
        U = simulate_utility_paths(alpha, pi_t, T, N, Msim, b_t, sigma_t, x0, F)
        plt.figure(figsize=(8,4))
        sel = np.random.default_rng(42).choice(Msim,5,replace=False)
        for i in sel:
            plt.plot(t, U[i], alpha=0.6)
        plt.xlabel('t'); plt.ylabel('U_t')
        plt.title(f'Paths of U_t (α={alpha})'); plt.grid(); plt.tight_layout(); plt.show()

# --- Main execution ------------------------------------------------------
def main():
    T, M = 1.0, 100_000
    b, sigma, x0 = 0.1, 0.2, 1.0
    alphas = [0.5, 1.0, 2.0, 5.0]

    for distribution, dist_params in scenarios:
        print(f"\n--- Scenario: {distribution} Liability ---")
        results = test_multiple_alphas(
            alphas, T, M, b, sigma, x0,
            distribution, dist_params, paths_seed=42
        )
        plot_combined_alpha(
            results, T, M, b, sigma, x0,
            distribution, dist_params
        )

    # sample-path utility plots
    sample_path_plots(alphas, T, Msim=5000, b=b, sigma=sigma, x0=x0)

    # time-varying strategy demo
    print("\n--- Time-Varying b(t), σ(t) Strategy ---")
    N = 200
    b_func     = lambda t: 0.1 + 0.05*np.sin(2*np.pi*t/T)
    sigma_func = lambda t: 0.2 + 0.1*np.cos(2*np.pi*t/T)
    plot_time_varying_strategy(alphas, T, N, b_func, sigma_func, x0)

if __name__ == "__main__":
    main()