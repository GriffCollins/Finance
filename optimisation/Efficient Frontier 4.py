import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import yfinance as yf
from sklearn.covariance import LedoitWolf
from datetime import datetime
import json

tickers = ["NEM", "PRIM", "LDO.MI", "TTWO", "ILMN", "ADYEN.AS", "DVN"]
risk_free_rate = 0.05
max_weight = 0.20
min_weight = 0.00

# Date configuration
lookback_start = "2023-12-01"  # 2 years before optimization date
lookback_end = "2025-12-01"  # Optimization date

#Download
print("=" * 60)
print("DOWNLOADING LOOKBACK DATA FOR OPTIMIZATION")
print("=" * 60)
print(f"Lookback period: {lookback_start} to {lookback_end}")

prices_lookback = yf.download(tickers, start=lookback_start, end=lookback_end, auto_adjust=True)["Close"]
returns_lookback = prices_lookback.pct_change().dropna()

print(f"Lookback data: {len(prices_lookback)} trading days")
print(f"Date range: {prices_lookback.index[0].date()} to {prices_lookback.index[-1].date()}")

print("\n" + "=" * 60)
print("CALCULATING OPTIMAL PORTFOLIO WEIGHTS")
print("=" * 60)

#Mean shrinkage
mu = returns_lookback.mean().values * 252
mu = 0.3 * mu + 0.7 * 0.10  # Shrink towards 10% return

# Ledoit-Wolf covariance estimation
lw = LedoitWolf().fit(returns_lookback.values)
cov_lw = lw.covariance_ * 252
n = len(mu)

print("\nExpected Returns (annualized, after shrinkage):")
for t, m in zip(tickers, mu):
    print(f"  {t:10s}: {m:7.2%}")

def portfolio_variance(w, cov):
    return w @ cov @ w

constraints = [
    {"type": "eq", "fun": lambda w: np.sum(w) - 1},  # Fully invested
]

bounds = [(min_weight, max_weight) for _ in range(n)]
w0 = np.ones(n) / n

# Minimum Variance
print("\n" + "=" * 60)
print("MINIMUM VARIANCE PORTFOLIO")
print("=" * 60)

res_mvp = minimize(
    portfolio_variance,
    w0,
    args=(cov_lw,),
    method="SLSQP",
    bounds=bounds,
    constraints=constraints,
    options={"ftol": 1e-9, "disp": False}
)

if not res_mvp.success:
    raise RuntimeError("MVP optimization failed")

w_mvp = res_mvp.x
ret_mvp = w_mvp @ mu
vol_mvp = np.sqrt(w_mvp @ cov_lw @ w_mvp)
sharpe_mvp = (ret_mvp - risk_free_rate) / vol_mvp

print(f"\nExpected Return: {ret_mvp:.2%}")
print(f"Expected Volatility: {vol_mvp:.2%}")
print(f"Expected Sharpe Ratio: {sharpe_mvp:.2f}")
print("\nOptimal Weights:")
for t, w in zip(tickers, w_mvp):
    if w > 0.001:
        print(f"  {t:10s}: {w:6.2%}")

# Efficient Frontier
print("\n" + "=" * 60)
print("COMPUTING EFFICIENT FRONTIER")
print("=" * 60)

target_returns = np.linspace(mu.min(), mu.max(), 40)
frontier_vol = []
frontier_weights = []

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
        constraints=cons,
        options={"ftol": 1e-9, "disp": False}
    )

    if res.success:
        frontier_vol.append(np.sqrt(res.fun))
        frontier_weights.append(res.x)
    else:
        frontier_vol.append(np.nan)
        frontier_weights.append(None)

# Plot Efficient Frontier
plt.figure(figsize=(12, 8))
plt.plot(frontier_vol, target_returns, 'b-', linewidth=2, label='Efficient Frontier')
plt.scatter([vol_mvp], [ret_mvp], color='green', s=200, marker='*',
            label=f'Min Variance (Sharpe={sharpe_mvp:.2f})', zorder=5)

# Plot individual assets
for i, ticker in enumerate(tickers):
    plt.scatter([np.sqrt(cov_lw[i, i])], [mu[i]], alpha=0.6, s=100)
    plt.annotate(ticker, (np.sqrt(cov_lw[i, i]), mu[i]),
                 xytext=(5, 5), textcoords='offset points', fontsize=8)

plt.xlabel("Volatility (Annualized)", fontsize=12)
plt.ylabel("Expected Return (Annualized)", fontsize=12)
plt.title("Efficient Frontier (Based on 2-Year Lookback)", fontsize=14, fontweight='bold')
plt.legend(loc='best')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('efficient_frontier.png', dpi=300, bbox_inches='tight')
print("Efficient Frontier plot saved as 'efficient_frontier.png'")

# -----------------------------
# Save Optimal Weights to File
# -----------------------------
weights_data = {
    'tickers': tickers,
    'weights': w_mvp.tolist(),
    'expected_return': float(ret_mvp),
    'expected_volatility': float(vol_mvp),
    'sharpe_ratio': float(sharpe_mvp),
    'optimization_date': lookback_end,
    'risk_free_rate': risk_free_rate
}

with open('optimal_weights.json', 'w') as f:
    json.dump(weights_data, f, indent=2)

print("\n" + "=" * 60)
print("WEIGHTS SAVED TO 'optimal_weights.json'")
print("=" * 60)
print("\nYou can now use these weights in the portfolio management tool.")

plt.show()
