from functions import *

def main():
    T, M = 1.0, 10_000
    b, sigma, alpha, x0 = 0.1, 0.2, 1.0, 1.0
    pi_grid = np.linspace(-4, 10, 100)
    W_T = simulate_WT(T, M, seed=42)

    pi_det, eu_det = find_optimal_pi(pi_grid, W_T, "constant", {"const_value": 0.5}, x0, b, sigma, alpha, seed=42)
    print(f"Optimal π (deterministic F) = {pi_det:.2f}")
    plot_utilities(pi_grid, eu_det, "Deterministic Liability F=0.5", pi_det)

    pi_rand, eu_rand = find_optimal_pi(pi_grid, W_T, "normal", {"normal_mu": 10.0, "normal_sigma": 2.0}, x0, b, sigma, alpha, seed=42)
    print(f"Optimal π (random F)      = {pi_rand:.2f}")
    plot_utilities(pi_grid, eu_rand, "Random Liability F~N(10,2²)", pi_rand)


if __name__ == "__main__":
    main()