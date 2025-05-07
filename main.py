from functions import *

def main():
    # Parameters
    T = 1.0
    M = 100_000  # Monte Carlo paths
    b, sigma, alpha, x0 = 0.1, 0.2, 1.0, 1.0
    # Analytic optimal pi
    analytic_pi = b / (alpha * sigma**2)

    # Focused grid around analytic value
    pi_grid = np.linspace(2.0, 3.0, 300)
    W_T = simulate_WT(T, M, seed=42)

    # Deterministic liability
    pi_det, eu_det = find_optimal_pi(
        pi_grid, W_T,
        "constant", {"const_value": 0.5},
        x0, b, sigma, alpha, T,
        seed=42
    )
    print(f"Analytic π* = {analytic_pi:.2f}")
    print(f"Numeric π* (deterministic F) = {pi_det:.2f}")
    plot_utilities(pi_grid, eu_det, "Deterministic Liability F=0.5", pi_det)

    # Random liability (skipped sampling to verify pi*)
    pi_rand, eu_rand = find_optimal_pi(
        pi_grid, W_T,
        "normal", {"normal_mu": 10.0, "normal_sigma": 2.0},
        x0, b, sigma, alpha, T,
        seed=42
    )
    print(f"Numeric π* (random F)      = {pi_rand:.2f}")
    plot_utilities(pi_grid, eu_rand, "Random Liability F~N(10,2²)", pi_rand)

if __name__ == "__main__":
    main()
