import numpy as np
import matplotlib.pyplot as plt
from functions import (
    simulate_WT,
    generate_F,
    expected_utility,
    expected_power_utility,
    expected_log_utility,
    find_optimal_pi
)
from typing import Optional
import pandas as pd

# --- CONFIGURATION --------------------------------------------------------
# Utility settings: 'exponential', 'power', or 'log'
UTILITY_TYPE = 'power'
ALPHAS = [0.5, 1.0, 2.0, 5.0]   # for exponential (α)
GAMMAS = [3, 5.0]   # for power (γ)

SCENARIOS = [
    ('constant',          {'const_value': 0.5}),
    ('normal',            {'normal_mu': 0.5, 'normal_sigma': 1.0}),
    ('hedgable_gaussian', {'mu_F': 0.5, 'sigma_F': 1.0}),
]
MC_SAMPLES = 500_000
GRID_PTS = 500
GRID_WINDOW_RATIO = 0.1  # ±10% around analytic

# --------------------------------------------------------------------------
def compute_analytic_pi(
    utility: str,
    dist: str,
    params: dict,
    risk_param: Optional[float],
    b: float,
    sigma: float,
    T: float,
    x0: float
) -> Optional[float]:
    """
    Compute the closed-form optimal π* for exponential, power, or log utility.
    Returns None if no simple analytic form is implemented.
    """
    theta = b / sigma

    if utility == 'exponential':
        alpha = risk_param
        if dist == 'hedgable_gaussian':
            kappa = params['sigma_F'] / np.sqrt(T)
            return kappa/sigma + b/(alpha * sigma**2)
        return b/(alpha * sigma**2)

    if utility == 'power':
        gamma = risk_param
        # γ=1 → reduce to log utility
        if abs(gamma - 1.0) < 1e-8:
            return b / sigma

        barF = params.get('const_value', params.get('normal_mu', 0.0))
        theta2 = theta**2
        # fixed sign in Y0
        Y0 = barF - (gamma/(2*(1-gamma))) * theta2 * T
        H0 = x0 - Y0
        base = (b/(sigma**2)) * (H0/(1-gamma))
        if dist == 'hedgable_gaussian':
            muF = params['mu_F']
            sigmaF = params['sigma_F']
            kappa = sigmaF / np.sqrt(T)
            base += kappa/sigma
        return base

    if utility == 'log':
        return theta

    return None

# --------------------------------------------------------------------------
def test_strategy(
    T: float,
    b: float,
    sigma: float,
    x0: float,
    distribution: str,
    dist_params: dict
):
    """
    For each risk parameter, compute analytic π* (if available)
    and either numeric π* (exp/log) or dynamic utility + MSE‐best π (power).
    """
    if UTILITY_TYPE == 'exponential':
        risk_list = ALPHAS
    elif UTILITY_TYPE == 'power':
        risk_list = GAMMAS
    else:
        risk_list = [None]

    # Pre-simulate W_T for exp/log numeric search
    W_T = simulate_WT(T, MC_SAMPLES, seed=42)
    results = []

    for rp in risk_list:
        analytic = compute_analytic_pi(
            UTILITY_TYPE, distribution, dist_params,
            rp, b, sigma, T, x0
        )

        if UTILITY_TYPE == 'exponential':
            # build π-grid around analytic
            center = analytic if analytic is not None else b/(sigma**2)
            half_width = GRID_WINDOW_RATIO * abs(center)
            pi_grid = np.linspace(center - half_width,
                                   center + half_width,
                                   GRID_PTS)

            pi_num, eu_vals = find_optimal_pi(
                pi_grid, W_T,
                distribution, dist_params,
                x0, b, sigma, rp, T,
                seed=42
            )
            print(f"{distribution} | exp α={rp:.2f}: "
                  f"analytic={analytic:.4f}, numeric={pi_num:.4f}")
            results.append((rp, analytic, pi_num, pi_grid, eu_vals))

        elif UTILITY_TYPE == 'power':
            # Build terminal liability F_vec
            M = W_T.shape[0]
            if distribution == 'hedgable_gaussian':
                muF, sigmaF = dist_params['mu_F'], dist_params['sigma_F']
                kappa = sigmaF / np.sqrt(T)
                F_vec = muF + kappa * W_T
            elif distribution == 'normal':
                F_vec = np.zeros(M)
            else:
                F_vec = generate_F(M, distribution, **dist_params, seed=42)

            # Run dynamic CRRA and collect π-matrix
            util_dyn, pi_mat = expected_power_utility(
                x0=x0,
                b=b,
                sigma=sigma,
                gamma=rp,
                T=T,
                distribution=distribution,
                dist_params=dist_params,
                r=0.0,
                n_steps=200,
                M=MC_SAMPLES,
                seed=42
            )

            # Compute MSE‐best constant π
            pi_flat = pi_mat.ravel()
            p_mse   = pi_flat.mean()
            mse_val = np.mean((pi_flat - p_mse)**2)

            print(
                f"{distribution} | power γ={rp:.2f}: "
                f"analytic π*={analytic:.4f}, "
                f"simulated E[U]={util_dyn:.4f}, "
                f"MSE‐best π={p_mse:.4f} (MSE={mse_val:.2e})"
            )
            # store (γ, analytic π, E[U], p_mse, mse)
            results.append((rp, analytic, util_dyn, p_mse, mse_val))

        else:  # logarithmic
            # build π-grid around analytic
            center = analytic if analytic is not None else b/(sigma**2)
            half_width = GRID_WINDOW_RATIO * abs(center)
            pi_grid = np.linspace(center - half_width,
                                   center + half_width,
                                   GRID_PTS)

            # build F_vec
            M = W_T.shape[0]
            if distribution == 'hedgable_gaussian':
                muF, sigmaF = dist_params['mu_F'], dist_params['sigma_F']
                kappa = sigmaF / np.sqrt(T)
                F_vec = muF + kappa * W_T
            elif distribution == 'normal':
                F_vec = np.zeros(M)
            else:
                F_vec = generate_F(M, distribution, **dist_params, seed=42)

            eu_vals = np.array([
                expected_log_utility(pi, W_T, F_vec, x0, b, sigma, T)
                for pi in pi_grid
            ])
            pi_num = pi_grid[np.argmax(eu_vals)]
            print(f"{distribution} | log: analytic π*={analytic:.4f}, numeric={pi_num:.4f}")
            results.append((rp, analytic, pi_num, pi_grid, eu_vals))

    return results

# --------------------------------------------------------------------------
def plot_results(
    results,
    T: float,
    b: float,
    sigma: float,
    x0: float,
    distribution: str,
    dist_params: dict
):
    """
    Plot MC utility curves and mark numeric/analytic π* on a common π axis.
    Only for exponential & log utilities.
    """
    pi_min = min(pi[0] for _, _, _, pi, _ in results)
    pi_max = max(pi[-1] for _, _, _, pi, _ in results)
    pi_common = np.linspace(pi_min, pi_max, GRID_PTS)

    W_T = simulate_WT(T, MC_SAMPLES, seed=123)
    # prepare F_vec
    if distribution == 'hedgable_gaussian':
        muF, sigmaF = dist_params['mu_F'], dist_params['sigma_F']
        kappa = sigmaF / np.sqrt(T)
        F_vec = muF + kappa * W_T
    elif distribution == 'normal':
        F_vec = np.zeros_like(W_T)
    else:
        F_vec = generate_F(len(W_T), distribution, **dist_params, seed=123)

    plt.figure(figsize=(8,5))
    for rp, analytic, pi_num, pi_grid, eu_vals in results:
        if UTILITY_TYPE == 'exponential':
            U_common = [
                expected_utility(pi, W_T, F_vec, x0, b, sigma, rp, T)
                for pi in pi_common
            ]
            label = f"MC α={rp:.2f}"
        else:  # log
            U_common = [
                expected_log_utility(pi, W_T, F_vec, x0, b, sigma, T)
                for pi in pi_common
            ]
            label = "MC log"

        plt.plot(pi_common, U_common, label=label)
        plt.axvline(pi_num, color='k', linestyle='--',
                    label=f"Numeric π* (param={rp})")
        if analytic is not None:
            plt.axvline(analytic, color='r', linestyle=':',
                        label=f"Analytic π* (param={rp})")

    plt.xlabel('π')
    plt.ylabel('Expected Utility')
    plt.title(f"{UTILITY_TYPE.title()} Utility — {distribution}")
    plt.legend(fontsize='small', ncol=2)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# --------------------------------------------------------------------------
if __name__ == '__main__':
    T, b, sigma, x0 = 1.0, 0.1, 0.2, 1.0

    all_results = {}
    csv_rows = []

    for distribution, dist_params in SCENARIOS:
        print(f"--- Scenario: {distribution} ---")
        results = test_strategy(T, b, sigma, x0, distribution, dist_params)
        all_results[distribution] = results

        # Collect CSV rows
        for entry in results:
            rp = entry[0]
            row = {
                'utility': UTILITY_TYPE,
                'distribution': distribution,
                'risk_param': rp,
            }
            if UTILITY_TYPE == 'power':
                _, analytic, util_dyn, p_mse, mse_val = entry
                row.update({
                    'analytic_pi': analytic,
                    'simulated_EU': util_dyn,
                    'mse_best_pi': p_mse,
                    'mse_value': mse_val
                })
            else:
                _, analytic, pi_num, _, _ = entry
                row.update({
                    'analytic_pi': analytic,
                    'numeric_pi': pi_num
                })
            csv_rows.append(row)

        # Only plot for exponential & log
        if UTILITY_TYPE in ('exponential', 'log'):
            plot_results(results, T, b, sigma, x0, distribution, dist_params)

    # # Write out CSV
    # df = pd.DataFrame(csv_rows)
    # df.to_csv('optimal_pi_results.csv', index=False, mode='a')
    # print("Saved results to optimal_pi_results.csv")
