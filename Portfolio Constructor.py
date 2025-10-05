import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import statsmodels.api as sm
from datetime import datetime
import seaborn
import yfinance as yf

stocklist = ["AAPL", "TSLA", "META", "AMZN", "NVDA", "MSFT", "PLTR", "GOOGL", "WBD", "F"]
df = yf.download(tickers=stocklist, auto_adjust=True)
returns_data = df["Close"].pct_change().dropna()

def portfolio_variance(weights, cov_matrix):
    # Matrix multiplication between weights transpose, covariance matrix and weights
    return weights.T @ cov_matrix @ weights


def negative_sharpe_ratio(weights, expected_returns, cov_matrix, risk_free_rate):
    # Return is dot product between weights and annualised mean returns
    portfolio_ret = np.dot(weights, expected_returns)

    # Volatility is sqrt of variance
    portfolio_vol = np.sqrt(portfolio_variance(weights, cov_matrix))

    # Negative Sharpe Equation
    return - (portfolio_ret - risk_free_rate) / portfolio_vol


def sortino_ratio(returns, risk_free_rate, target_return=0):
    """
    Calculate the annualized Sortino ratio for a return series.
    """
    # Convert to excess returns over target
    excess_returns = returns - target_return

    # Filter only downside returns
    downside_returns = excess_returns[excess_returns < 0]

    # Downside deviation (annualized)
    downside_deviation = np.sqrt((downside_returns ** 2).mean()) * np.sqrt(252)

    # Mean excess return (annualized)
    mean_excess_return = excess_returns.mean() * 252

    if downside_deviation == 0:
        return np.nan  # Avoid division by zero

    return mean_excess_return / downside_deviation


# Weights must sum to 0 in function
def weight_constraint(weights):
    return np.sum(weights) - 1


def sortino_ratio(weights, expected_returns, returns_data, risk_free_rate=0.05, annualized_factor=252):
    """
    Calculate the Sortino Ratio for a portfolio.

    Parameters:
    - weights: Portfolio weights (array-like)
    - expected_returns: Annualized expected returns (array-like)
    - returns_data: Daily returns (DataFrame)
    - risk_free_rate: Annualized risk-free rate (float)
    - annualized_factor: Scaling factor (e.g., 252 for daily returns)

    Returns:
    - Sortino Ratio (float)
    """
    # Calculate portfolio returns
    portfolio_returns = np.dot(returns_data, weights)

    # Annualized mean return
    annualized_return = np.mean(portfolio_returns) * annualized_factor

    # Downside deviation: only returns below risk-free rate (daily)
    downside_returns = np.minimum(portfolio_returns - (risk_free_rate / annualized_factor), 0)
    downside_deviation = np.sqrt(np.mean(downside_returns ** 2)) * np.sqrt(annualized_factor)

    # Avoid division by zero
    if downside_deviation == 0:
        return np.inf if (annualized_return - risk_free_rate) > 0 else -np.inf

    # Sortino Ratio
    sortino = (annualized_return - risk_free_rate) / downside_deviation
    return sortino


def negative_sortino_ratio(weights, expected_returns, returns_data, risk_free_rate=0.05):
    """Wrapper for optimization (minimizing negative Sortino)."""
    return -sortino_ratio(weights, expected_returns, returns_data, risk_free_rate)

#Choose the risk-free rate
risk_free_rate = 0.05

#Make covariance matrix
cov_matrix = returns_data.cov()

#Annualise the covariance matrix
cov_matrix = cov_matrix * 252

#Create the annualised mean returns
expected_returns = returns_data.mean() * 252

#Measure the number of assets
num_assets = len(expected_returns)

#Retrieve ticker names
tickers = expected_returns.index

# The minimum and maximum allowable weights for your stock allocation
bounds = [(-1, 1) for _ in range(num_assets)]

#Create target returns vector so the maths can be repeated for each value for the graph
target_returns = np.linspace(0.05, 0.60, 100)

#Make the lists
efficient_vols = []
efficient_weights = []

#For loop for generating efficient frontier y coords  
for target_return in target_returns:
    #Return constraint "eq" means equality (other version is inequality) and also defines a function lambda
    return_constraint = {
        'type': 'eq',
        'fun': lambda w, mu=expected_returns, tr=target_return: np.dot(w, mu) - tr
    }

    #Add the return constraint to a list of dictionaries with a dictionary with referenced the weight_constraint function
    constraints = [return_constraint, {'type': 'eq', 'fun': weight_constraint}]
    
    #Calculate the initial weights
    initial_weights = np.ones(num_assets) / num_assets

    #Scipy minimise function using sequencial least squares programming
    result = minimize(
        fun=portfolio_variance,
        x0=initial_weights,
        args=(cov_matrix,),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )

    #The minimise function returns a class hence the OOP
    if result.success:
        #Retrieve the variance using result.x (the return)
        var = portfolio_variance(result.x, cov_matrix)

        #Append the volatility from the minimise to efficient volatility list
        efficient_vols.append(np.sqrt(var))

        #Append the weights to the weights list
        efficient_weights.append(result.x)
    else:
        #Else add nothing
        efficient_vols.append(np.nan)
        efficient_weights.append([np.nan] * num_assets)

#Plot the efficient frontier
plt.figure()
plt.plot(efficient_vols, target_returns, 'b--', linewidth=2)
plt.xlabel('Annualized Volatility (Risk)')
plt.ylabel('Expected Return')
plt.title('Efficient Frontier (Markowitz)')
plt.grid(True)
plt.show()

"""~Portfolio Allocation and Measurements~"""

#The function requires initial weights so 1/n
initial_weights = np.ones(num_assets) / num_assets

#Minimised variance portfolio calculation
mvp_result = minimize(
    fun=portfolio_variance,
    x0=initial_weights,
    args=(cov_matrix,),
    method='SLSQP',
    bounds=bounds,
    constraints=[{'type': 'eq', 'fun': weight_constraint}]
)

#Same as before
if mvp_result.success:

    #Assign weights
    mvp_weights = mvp_result.x

    #Assign volatility
    mvp_vol = np.sqrt(portfolio_variance(mvp_weights, cov_matrix))

    #Assign returns
    mvp_ret = np.dot(mvp_weights, expected_returns)

    #Calculate the sharpe
    mvp_sharpe = (mvp_ret - risk_free_rate) / mvp_vol
    
    #Print all the data, .2% is a string formatting tool for percentages
    print("\n Minimum Variance Portfolio:")
    print(f"Expected Return: {mvp_ret:.2%}")
    print(f"Volatility (Risk): {mvp_vol:.2%}")
    print(f"Sharpe Ratio: {mvp_sharpe:.2f}")
    print("Weights:")
    for ticker, weight in zip(tickers, mvp_weights):
        print(f"  {ticker}: {weight:.2%}")
else:
    print("MVP Optimization failed.")

#Max sharpe calculation
sharpe_result = minimize(
    fun=negative_sharpe_ratio,
    x0=initial_weights,
    args=(expected_returns, cov_matrix, risk_free_rate),
    method='SLSQP',
    bounds=bounds,
    constraints=[{'type': 'eq', 'fun': weight_constraint}]
)

#Same as before, but no more comments
if sharpe_result.success:
    sharpe_weights = sharpe_result.x
    sharpe_ret = np.dot(sharpe_weights, expected_returns)
    sharpe_vol = np.sqrt(portfolio_variance(sharpe_weights, cov_matrix))
    sharpe_ratio = (sharpe_ret - risk_free_rate) / sharpe_vol

    print("\n Maximum Sharpe Ratio Portfolio (Tangency Portfolio):")
    print(f"Expected Return: {sharpe_ret:.2%}")
    print(f"Volatility (Risk): {sharpe_vol:.2%}")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    print("Weights:")
    for ticker, weight in zip(tickers, sharpe_weights):
        print(f"  {ticker}: {weight:.2%}")
else:
    print("Sharpe Optimization failed.")

sortino_result = minimize(
    fun=negative_sortino_ratio,
    x0=initial_weights,
    args=(expected_returns, returns_data, risk_free_rate),
    method='SLSQP',
    bounds=bounds,
    constraints=[{'type': 'eq', 'fun': weight_constraint}]
)

if sortino_result.success:
    sortino_weights = sortino_result.x
    sortino_ret = np.dot(sortino_weights, expected_returns)
    sortino_vol = np.sqrt(portfolio_variance(sortino_weights, cov_matrix))
    sortino_ratio_value = sortino_ratio(sortino_weights, expected_returns, returns_data, risk_free_rate)

    print("\n Maximum Sortino Ratio Portfolio:")
    print(f"Expected Return: {sortino_ret:.2%}")
    print(f"Volatility (Risk): {sortino_vol:.2%}")
    print(f"Sortino Ratio: {sortino_ratio_value:.2f}")
    print("Weights:")
    for ticker, weight in zip(tickers, sortino_weights):
        print(f"  {ticker}: {weight:.2%}")
else:
    print("Sortino Optimization failed.")












    
    