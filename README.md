# util-maximization
Link to recording of meeting with the Professor.
Note: Access requires a Berkeley email.


## Overview
This repository explores exponential utility maximization in both simulated and real‐data settings. We start with a one‐period Black–Scholes model and constant liability, build up Monte‐Carlo and closed‐form comparisons, extend to multiple risk‐aversion parameters, simulate dynamic BSDE–style sample paths, then conclude with a time‐varying strategy applied to real NVDA historical returns.

### Core problem
An investor with exponential utility $U(x) = -e^{-\alpha x}$ allocates a constant position $\pi$ in one risky asset $S$ over horizon $[0, T]$, faces terminal liability $F$, and seeks
$$ \max_{\pi} E[U(X_T^{\pi} - F)] $$
Asset dynamics (Black–Scholes):
$$ dS_t = \mu S_t dt + \sigma S_t dW_t $$
Wealth at maturity:
$$ X_T^{\pi} = x_0 + \pi \int_0^T \frac{dS_t}{S_t} = x_0 + \pi(\mu T + \sigma W_T) $$
Liability: constant or random $F$. If random, we may assume $F \sim N(m_F, s_F^2)$.


## 🚀 Installation
Clone the repo:

```bash
git clone <this repo URL>
cd util-maximization
```

Install required packages:
```bash
pip install -r requirements.txt
```

### Simulation toolkit (functions.py & main.py)
One-step Monte Carlo (constant liability)
simulate_WT(T,M): Draws $M$ samples of $W_T$.
generate_F(...): Generates liability vector $F_{\text{vec}}$ (constant or normal).
expected_utility(π, W_T, F_vec, x0,b,σ,α,T): Estimates $E[U(X_T^{\pi} - F)]$ for fixed $\pi$.
find_optimal_pi(...): Grid‐search over $\pi$ to find numeric optimum.
plot_utilities(...): Visualizes expected utility vs. $\pi$.
Run python main.py to:
Confirm analytic optimum $\pi^* = \frac{\mu}{\alpha \sigma^2} - \frac{\text{Cov}(W_T, F)}{T \sigma}$ for constant and normal liabilities.
Sweep across multiple $\alpha$ values and plot both analytic vs numeric $\pi^*$ vs $\alpha$.
Show Monte‐Carlo utility‐vs‐$\pi$ curves side‐by‐side.
Simulate BSDE‐style sample paths of $X_t$ under optimal $\pi$.
Comparison to closed‐form For constant or Gaussian liabilities, we have a closed‐form certainty equivalent $$ CE(\pi) = E[X_T^{\pi} - F] - \frac{\alpha}{2} \text{Var}(X_T^{\pi} - F) $$ so that $$ E[U(X_T^{\pi} - F)] = -\exp(-\alpha CE(\pi)) $$ We overlay these analytic utility curves (dotted) with Monte‐Carlo estimates (solid) to confirm near‐perfect agreement.
### 📈 Time‐Varying Parameters Demo
We then illustrate how a dynamic strategy adapts when $\mu$ and $\sigma$ change over time.

plot_time_varying_strategy:
Plots $\pi_t^* = \frac{\mu_t}{\alpha_t \sigma_t^2}$ for various $\alpha_t$.
Simulates and displays sample paths of $X_t$ under this policy.
### 📊 Real‐Data Back‐Test (NVDA)
Finally, we apply the framework to 5 years of NVIDIA daily prices (CSV: nvda_historical_data.csv).

Load CSV and parse Close prices.
Compute daily returns and rolling $\mu_t, \sigma_t$ over a 20‐day window, annualized.
Dynamic policy: $\pi_t^* = \frac{\mu_t}{\alpha \sigma_t^2}$. Plot for $\alpha_t = \alpha$.
Back‐test for a given $\alpha$: propagate wealth $X_0 \rightarrow X_T$ and map to utility $U(X_T - F)$.
Static benchmark: use $E[\mu], E[\sigma^2]$ to compute closed‐form $\pi_{\text{analytic}}^*$.
Compare back‐test vs analytic $\pi^*$ in a two‐panel figure.
Interpretation
Spikes in $\pi^*$ reflect rapid changes in estimated drift/volatility.
Back‐test utility outperforms or underperforms the static benchmark depending on realized returns vs long‐term average.