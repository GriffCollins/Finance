import yfinance as yf
import numpy as np
from numpy.random import Generator, PCG64
import matplotlib.pyplot as plt

# Portfolio setup
stocklist = ["AAPL", "TSLA", "META", "AMZN", "NVDA", "MSFT", "PLTR", "GOOGL", "WBD", "F"]
weights = np.array([-0.2257, -0.0071, 1.00, -0.1089, 0.0676, 0.1159, 0.7019, -0.0846, -0.0546, -0.4044])

# Download data
df = yf.download(stocklist, period="5y", auto_adjust=True)
returns = df["Close"].pct_change().dropna()
daily_std = np.std(returns, ddof=1)
daily_mean = returns.mean()

# Monte Carlo simulation
rng = Generator(PCG64(seed=42))
num_simulations = 100000
simulation_days = 252
portfolio_returns = np.zeros(num_simulations)  # Store final cumulative returns

cov_matrix = returns.cov().values  # shape: (num_stocks, num_stocks)

# Step 2: Cholesky decomposition
cholesky_matrix = np.linalg.cholesky(cov_matrix)


for sim in range(num_simulations):
    # Simulate daily returns for all stocks independently
    simulated_returns = np.zeros((simulation_days, len(stocklist)))

    z = rng.normal(size=(simulation_days, len(stocklist)))  # uncorrelated standard normals
    correlated_returns = z @ cholesky_matrix.T  # inject correlation
    simulated_returns = correlated_returns + daily_mean.values

    # Calculate daily portfolio returns (accounting for short positions)
    daily_portfolio_returns = np.dot(simulated_returns, weights.T)

    # Cumulative return over the simulation period
    cumulative_return = np.prod(1 + daily_portfolio_returns) - 1
    portfolio_returns[sim] = cumulative_return

# Calculate VaR directly from the simulated portfolio returns
confidence_level = 0.95
VaR = -np.percentile(portfolio_returns, 100 * (1 - confidence_level))

#Expected Shortfall - CVaR
tail = portfolio_returns[portfolio_returns <= -VaR]
cvar = -tail.mean()


print(f"Monte Carlo 95% VaR: {VaR:.2%}")
print(f"Monte Carlo Expected Shortfall: {cvar:.2%}")

# Plot histogram of simulated returns
plt.figure(figsize=(10, 6))
plt.hist(portfolio_returns, bins=50, alpha=0.75, color='blue')
plt.axvline(-VaR, color='red', linestyle='--', label=f'95% VaR: {-VaR:.2%}')
plt.title("Distribution of Simulated Portfolio Returns (1 Year)")
plt.xlabel("Portfolio Return")
plt.ylabel("Frequency")
plt.legend()
plt.show()







