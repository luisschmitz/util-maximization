import numpy as np
import matplotlib.pyplot as plt
from functions import (
    simulate_WT, generate_F,
    expected_utility, expected_power_utility,
    find_optimal_pi
)
from typing import Optional

# --- CONFIGURATION --------------------------------------------------------
# Utility settings: 'exponential' or 'power'
UTILITY_TYPE = 'exponential'
ALPHAS = [0.5, 1.0, 2.0, 5.0]   # for exponential (α)
GAMMAS = [0.5, 1.0, 2.0, 5.0]   # for power (γ)

# Liability scenarios
SCENARIOS = [
    ('constant',          {'const_value': 0.5}),
    ('normal',            {'normal_mu': 0.5, 'normal_sigma': 1.0}),
    ('hedgable_gaussian', {'mu_F': 0.5,     'sigma_F': 1.0}),
]

# Monte-Carlo sample size (increased for accuracy)
MC_SAMPLES = 500_000

# Grid-search settings: narrower around analytic π*
GRID_PTS = 500
GRID_WINDOW_RATIO = 0.25  # ±25% around analytic

# --------------------------------------------------------------------------
def compute_analytic_pi(
    utility: str,
    dist: str,
    params: dict,
    risk_param: float,
    b: float,
    sigma: float,
    T: float,
    x0: float
) -> Optional[float]:
    theta = b / sigma
    if utility == 'exponential':
        alpha = risk_param
        if dist == 'hedgable_gaussian':
            kappa = params['sigma_F'] / np.sqrt(T)
            return kappa/sigma + b/(alpha * sigma**2)
        return b/(alpha * sigma**2)
    if utility == 'power':
        gamma = risk_param
        if abs(gamma - 1.0) < 1e-8:
            return None
        barF = params.get('const_value', params.get('normal_mu', 0.0))
        theta2 = theta**2
        # Y0 contribution for constant or normal
        Y0 = barF + (gamma/(2*(1-gamma))) * theta2 * T
        H0 = x0 - Y0
        base = (b/(sigma**2)) * (H0/(1-gamma))
        if dist == 'hedgable_gaussian':
            muF = params['mu_F']; sigmaF = params['sigma_F']
            kappa = sigmaF / np.sqrt(T)
            # additive hedge term
            base += kappa/sigma
        return base
    return None

# --------------------------------------------------------------------------
def test_strategy(
    risk_params,
    T, b, sigma, x0,
    distribution, dist_params
):
    # choose correct list of risk parameters
    params = ALPHAS if UTILITY_TYPE=='exponential' else GAMMAS
    W_T = simulate_WT(T, MC_SAMPLES, seed=42)
    results = []
    for rp in params:
        analytic = compute_analytic_pi(
            UTILITY_TYPE, distribution, dist_params,
            rp, b, sigma, T, x0
        )
        # center grid at analytic or Merton baseline
        center = analytic if analytic is not None else b/(sigma**2)
        width = GRID_WINDOW_RATIO * abs(center)
        pi_grid = np.linspace(center-width, center+width, GRID_PTS)
        # numeric search
        if UTILITY_TYPE=='exponential':
            pi_num, eu_vals = find_optimal_pi(
                pi_grid, W_T,
                distribution, dist_params,
                x0, b, sigma, rp, T, seed=42
            )
        else:
            # simulate liabilities
            M = W_T.shape[0]
            if distribution=='hedgable_gaussian':
                muF = dist_params['mu_F']; sigmaF = dist_params['sigma_F']
                kappa = sigmaF/np.sqrt(T)
                F_vec = muF + kappa*W_T
            elif distribution=='normal':
                F_vec = np.zeros(M)
            else:
                F_vec = generate_F(M, distribution, **dist_params, seed=42)
            # evaluate power utility with GBM model
            eu_vals = np.array([
                expected_power_utility(pi, W_T, F_vec,
                                        x0, b, sigma, rp, T)
                for pi in pi_grid
            ])
            pi_num = pi_grid[eu_vals.argmax()]
        print(f"{distribution} | {UTILITY_TYPE} param={rp}: analytic={analytic}, numeric={pi_num:.2f}")
        results.append((rp, analytic, pi_num, pi_grid, eu_vals))
    return results

# --------------------------------------------------------------------------
def plot_results(results, T, b, sigma, x0, distribution, dist_params):
    """
    Overlay MC utility curves for each risk parameter on a common π-grid,
    and mark numeric and analytic π* for each.
    """
    # Build a common π-axis spanning all individual grids
    pi_min = min(pi_grid[0] for _, _, _, pi_grid, _ in results)
    pi_max = max(pi_grid[-1] for _, _, _, pi_grid, _ in results)
    pi_common = np.linspace(pi_min, pi_max, GRID_PTS)

    # Prepare the liability vector F_vec once per scenario
    W_T = simulate_WT(T, MC_SAMPLES, seed=123)
    if distribution == 'hedgable_gaussian':
        muF, sigmaF = dist_params['mu_F'], dist_params['sigma_F']
        kappa = sigmaF / np.sqrt(T)
        F_vec = muF + kappa * W_T
    elif distribution == 'normal':
        F_vec = np.zeros_like(W_T)
    else:  # constant
        F_vec = generate_F(len(W_T), distribution, **dist_params, seed=123)

    plt.figure(figsize=(8, 5))
    # Plot Monte Carlo curves
    for rp, analytic, pi_num, _, _ in results:
        if UTILITY_TYPE == 'exponential':
            U_common = [expected_utility(pi, W_T, F_vec, x0, b, sigma, rp, T)
                        for pi in pi_common]
            label = f"MC α={rp:.2f}"
        else:
            U_common = [expected_power_utility(pi, W_T, F_vec, x0, b, sigma, rp, T)
                        for pi in pi_common]
            label = f"MC γ={rp:.2f}"
        plt.plot(pi_common, U_common, label=label)

    # Mark numeric and analytic optima
    for rp, analytic, pi_num, _, _ in results:
        plt.axvline(pi_num, color='k', linestyle='--',
                    label=f"Numeric π* (param={rp:.2f})")
        if analytic is not None:
            plt.axvline(analytic, color='r', linestyle=':',
                        label=f"Analytic π* (param={rp:.2f})")

    plt.xlabel('π')
    plt.ylabel('Expected Utility')
    plt.title(f"{UTILITY_TYPE.title()} Utility — {distribution}")
    plt.legend(fontsize='small', ncol=2)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# --------------------------------------------------------------------------
if __name__=='__main__':
    # model parameters
    T, b, sigma, x0 = 1.0, 0.1, 0.2, 1.0
    for dist, dp in SCENARIOS:
        print(f"\n--- Scenario: {dist} ---")
        res = test_strategy(
            None, T, b, sigma, x0, dist, dp
        )
        plot_results(res, T, b, sigma, x0, dist, dp)