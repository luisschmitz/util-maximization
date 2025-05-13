# util-maximization

## Overview
This repository explores utility maximization for different investor preferences under both simulated and real-data settings. We study exponential, power, and logarithmic utility functions in the presence of terminal liabilities—both constant and random. Our analysis starts with a one-period Black–Scholes model and compares closed-form and Monte Carlo methods. We then simulate dynamic BSDE-style strategies and conclude with a real-data backtest using NVIDIA (NVDA) stock prices.

The theoretical foundation is based on Hu, Imkeller, and Müller's (2005) BSDE framework for utility maximization in incomplete markets. A full technical write-up with proofs, derivations, and extensions is included in `INDENG_222__Project.pdf`.

### Core Problem
We consider an investor with a utility function $U(x) = f(x)$, trading a risky asset $S$ over horizon $[0,T]$, facing terminal liability $F$, and solving:
$$\max_{\pi} \mathbb{E}[U(X_T^\pi - F)]$$

Asset dynamics follow the Black–Scholes model:
$$dS_t = \mu S_t\,dt + \sigma S_t\,dW_t$$

Wealth at maturity:
$$X_T^\pi = x_0 + \pi(\mu T + \sigma W_T)$$

Liability $F$ can be constant or normally distributed, e.g. $F \sim \mathcal{N}(\mu_F, \sigma_F^2)$.

We analyze this setup under three utility functions:
1. **Exponential Utility**: $U(x) = -e^{-\alpha x}$
2. **Power Utility**: $U(x) = \frac{x^\gamma}{\gamma}$
3. **Logarithmic Utility**: $U(x) = \log(x)$

## 📦 Installation

```bash
git clone <this-repo-URL>
cd util-maximization
pip install -r requirements.txt
```

## 🔧 Simulation Toolkit (functions.py & main.py)

Key components:
- `simulate_WT(T, M)`: Simulates Brownian terminal values.
- `generate_F(...)`: Creates liabilities (constant or normal).
- `expected_utility(...)`: Computes expected utility for fixed $\pi$.
- `find_optimal_pi(...)`: Searches for optimal $\pi^*$ via Monte Carlo.
- `plot_utilities(...)`: Plots utility-vs-$\pi$ comparisons.
- `simulate_expected_power_utility_rho(...)`: Power utility with dynamic hedging ratio $\rho$.

Run `python main.py` to:
- Recover analytic and numeric $\pi^*$ for exponential utility under constant and Gaussian liabilities.
- Extend simulations to power and log utility.
- Visualize utility landscapes and validate closed-form vs. simulation.
- Simulate BSDE-style strategies.

## 📘 Analytical Comparisons

For exponential utility and Gaussian liabilities:
$$E[U(X_T^\pi-F)] = -\exp(-\alpha(E[X_T^\pi-F]-\frac{\alpha}{2}Var(X_T^\pi-F)))$$

We overlay this closed-form expression with Monte Carlo estimates to validate our numerical pipeline.

Analytical results for power and log utilities are derived in specific cases and compared to simulated estimates.

## ⏱ Time-Varying Strategy

We explore the case where model parameters vary over time:
$$\pi_t^* = \mu_t/(\alpha_t\sigma_t^2)$$

- Plots sample paths of optimal wealth with rolling drift/volatility inputs.

## 📊 Real-Data Backtest (NVDA)

Using `nvda_historical_data.csv`, we:
- Parse 5 years of daily Close prices.
- Estimate rolling $\mu_t, \sigma_t$ over a 20-day window (annualized).
- Compute and plot time-varying $\pi_t^*$ for a fixed $\alpha$.
- Back-test portfolio growth using the dynamic strategy.
- Benchmark against static $\pi^*$ derived from long-run averages.

## 📄 Paper Summary

The theoretical basis for this project is drawn from:
- Hu, Imkeller, and Müller (2005): Utility Maximization in Incomplete Markets

Their framework uses BSDEs to derive optimal trading strategies under general utility functions and trading constraints.

We replicate their methods and:
- Introduce bounded liabilities for all three utility types.
- Derive explicit strategies and hedging terms under constant and hedgeable Gaussian liabilities.
- Numerically simulate expected utility and verify performance of the BSDE-based policies.

See `INDENG_222__Project.pdf` in the repository for a full write-up including all derivations, figures, and interpretation.
