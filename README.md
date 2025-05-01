# util-maximization

Link to [recording of meeting](https://drive.google.com/drive/folders/17xSSZBhEHa0HgjLo4sHVny9m2aX83fOj?usp=share_link) with the Professor.
 - Note: Can only access this with Berkeley email.


## Utility Maximization Simulation
A small Python toolkit to numerically solve the “exponential utility + constant liability” problem in a one-period Black–Scholes market via Monte Carlo.

📦 Installation
Clone this repo or copy the script.

Install dependencies: `pip install -r requirements.txt`

### Overview

We consider an investor with exponential utility  
$$
U(x) = -\exp\bigl(-\alpha\,x\bigr)
$$
who invests a constant amount \(\pi\) in a single risky asset over one period \(T = 1\), faces a terminal liability \(F\), and seeks to maximize  
$$
\mathbb{E}\bigl[\,U(X_T - F)\bigr].
$$

- **Asset dynamics:**  
  $$
    \frac{dS_t}{S_t} = b\,dt + \sigma\,dW_t.
  $$

- **Wealth at maturity:**  
  $$
    X_T = x_0 + \pi\,\bigl(b\,T + \sigma\,W_T\bigr).
  $$

- **Liability:**  
  Constant \(F\) (e.g.\ \(F = 0.5\)).

- **Simulation:**  
  Monte Carlo with \(M\) draws of  
  \[
    W_T \sim \mathcal{N}(0,\,T).
  \]

#### Current Assumptions

- One-period model with horizon \(T = 1\).  
- Static investment strategy \(\pi\).  
- Risky asset drift \(b\) and volatility \(\sigma\).  
- Liability \(F\) known and identical across paths (here \(F = 0.5\)).  
- Utility is exponential with risk-aversion \(\alpha\).  
- Monte Carlo: \(M\) i.i.d. draws of \(W_T\sim\mathcal{N}(0,\,T)\), fixed seed for reproducibility.  
