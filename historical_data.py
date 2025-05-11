import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1) Load and clean NVDA data
df = pd.read_csv(
    'nvda_historical_data.csv',
    parse_dates=['Date'],
    index_col='Date'
)
prices = (
    df['Close']
      .astype(str)
      .str.replace(r'[\$,]', '', regex=True)
      .astype(float)
      .rename('Price')
)
returns = prices.pct_change().dropna()

# 2) Compute rolling drift & vol
window = 20
mu_r    = returns.rolling(window).mean().dropna()
sigma_r = returns.rolling(window).std().dropna()
b_t     = mu_r * 252
sigma_t = sigma_r * np.sqrt(252)

# 3) Pick α and backtest dynamic strategy
alpha = 1.0
pi_t = b_t / (alpha * sigma_t**2)
pi_t = pi_t.reindex(returns.index).fillna(method='bfill')

x0 = 1.0
# backtest wealth
X = pd.Series(index=returns.index, dtype=float)
X.iloc[0] = x0
for i in range(1, len(returns)):
    X.iloc[i] = X.iloc[i-1] * (1 + pi_t.iloc[i-1] * returns.iloc[i])

# compute realized utility path
U_real = -np.exp(-alpha * (X - 0.0))

# 4) Compute analytic closed-form using constant b̄,σ̄
T_days = len(returns)
T_years = T_days / 252
t_frac = np.arange(T_days) / 252  # fraction of year
b_bar     = b_t.mean()
sigma_bar = sigma_t.mean()
pi_star   = b_bar / (alpha * sigma_bar**2)

# deterministic wealth path: X_ana(t) = x0 + π* · b̄ · t
X_ana = x0 + pi_star * b_bar * t_frac
U_ana = -np.exp(-alpha * (X_ana - 0.0))

# 5) Plot comparison
fig, ax = plt.subplots(2,1, figsize=(10,8), sharex=True)

ax[0].plot(returns.index, X,     label='Backtest Wealth $X_t$')
ax[0].plot(returns.index, X_ana, '--', label='Analytic Wealth $X_t^{\\rm ana}$')
ax[0].set_ylabel('Wealth')
ax[0].legend(); ax[0].grid(True)

ax[1].plot(returns.index, U_real, label='Backtest Utility $U_t$')
ax[1].plot(returns.index, U_ana, '--', label='Analytic Utility $U_t^{\\rm ana}$')
ax[1].set_ylabel('Utility')
ax[1].set_xlabel('Date')
ax[1].legend(); ax[1].grid(True)

plt.suptitle(f'Backtest vs Analytic Closed-Form (α={alpha}) for NVDA')
plt.tight_layout(rect=[0,0.03,1,0.95])
plt.show()
