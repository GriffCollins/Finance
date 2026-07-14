import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import yfinance as yf
from sklearn.covariance import LedoitWolf

tickers = ["NEM", "PRIM", "LDO.MI", "TTWO", "ILMN", "ADYEN.AS", "DVN"]
prices = yf.download(tickers, auto_adjust=True)["Close"]
returns = prices.pct_change().dropna()
mu = returns.mean().values * 252
mu = 0.3 * mu + 0.7 * 0.10
risk_free_rate = 0.05

# Fit Ledoit–Wolf on daily returns
lw = LedoitWolf().fit(returns.values)

# Annualised shrunk covariance matrix
cov_lw = lw.covariance_ * 252
n = len(mu)

def portfolio_variance(w, cov):
    return w @ cov @ w

def neg_sharpe(w, mu, cov, rf):
    ret = w @ mu
    vol = np.sqrt(w @ cov @ w)
    return -(ret - rf) / vol

constraints = [
    {"type": "eq", "fun": lambda w: np.sum(w) - 1},          # Fully invested
]

# Simple, implementable rules
max_weight = 0.20     # concentration cap
min_weight = 0.00     # long-only
bounds = [(min_weight, max_weight) for _ in range(n)]

# Initial guess
w0 = np.ones(n) / n

#Max Sharpe
res = minimize(
    neg_sharpe,
    w0,
    args=(mu, cov_lw, risk_free_rate),
    method="SLSQP",
    bounds=bounds,
    constraints=constraints,
    options={"ftol": 1e-9, "disp": False}
)

if not res.success:
    raise RuntimeError("Optimization failed")

w_star = res.x
ret_star = w_star @ mu
vol_star = np.sqrt(w_star @ cov_lw @ w_star)
sharpe_star = (ret_star - risk_free_rate) / vol_star

print("\nMax Sharpe Portfolio")
print(f"Return: {ret_star:.2%}")
print(f"Volatility: {vol_star:.2%}")
print(f"Sharpe: {sharpe_star:.2f}")
print("Weights:")
for t, w in zip(tickers, w_star):
    print(f"{t}: {w:.2%}")

target_returns = np.linspace(mu.min(), mu.max(), 40)
frontier_vol = []

for tr in target_returns:
    cons = constraints + [
        {"type": "eq", "fun": lambda w, tr=tr: w @ mu - tr}
    ]

    res = minimize(
        portfolio_variance,
        w0,
        args=(cov_lw,),
        method="SLSQP",
        bounds=bounds,
        constraints=cons
    )

    if res.success:
        frontier_vol.append(np.sqrt(res.fun))
    else:
        frontier_vol.append(np.nan)

plt.plot(frontier_vol, target_returns)
plt.xlabel("Volatility")
plt.ylabel("Return")
plt.title("Efficient Frontier")
plt.grid()
plt.show()

def portfolio_variance(w, cov):
    return w @ cov @ w

# constraints
constraints = [
    {"type": "eq", "fun": lambda w: np.sum(w) - 1}
]

# long-only + concentration control
max_weight = 0.20
bounds = [(0.0, max_weight) for _ in range(len(mu))]

# initial guess
w0 = np.ones(len(mu)) / len(mu)

#Minimum Variance
res = minimize(
    portfolio_variance,
    w0,
    args=(cov_lw,),
    method="SLSQP",
    bounds=bounds,
    constraints=constraints
)

if not res.success:
    raise RuntimeError("MVP optimisation failed")

# MVP weights
w_mvp = res.x

# Ex-post measurements
mvp_return = w_mvp @ mu
mvp_vol = np.sqrt(w_mvp @ cov_lw @ w_mvp)
mvp_sharpe = (mvp_return - risk_free_rate) / mvp_vol

print("\nMinimum Variance Portfolio (ex-post metrics)")
print(f"Expected Return: {mvp_return:.2%}")
print(f"Volatility: {mvp_vol:.2%}")
print(f"Sharpe Ratio: {mvp_sharpe:.2f}")

print("Weights:")
for t, w in zip(tickers, w_mvp):
    print(f"{t}: {w:.2%}")

port_rets = returns @ w_mvp
realised_return = port_rets.mean() * 252
realised_vol = port_rets.std() * np.sqrt(252)
realised_sharpe = (realised_return - risk_free_rate) / realised_vol
print(port_rets, realised_return, realised_vol, realised_sharpe)

