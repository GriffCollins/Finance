import numpy as np
import yfinance as yf
import pandas as pd

stocklist = ["AAPL", "TSLA", "META", "AMZN", "NVDA", "MSFT", "PLTR", "GOOGL", "WBD", "F"]
weights = np.array([-0.2257, -0.0071, 1.00, -0.1089, 0.0676, 0.1159, 0.7019, -0.0846, -0.0546, -0.4044])
window = 30
z_score = 1.96
position = 100000

df = pd.concat({ticker: yf.download(ticker, period='1y', auto_adjust=True)['Close'] for ticker in stocklist},axis=1)

log_returns = np.log(df).diff().dropna()
covariance_matrix = log_returns.rolling(window).cov(pairwise=True).dropna()
latest_covariance = log_returns.tail(window).cov()
portfolio_variance = np.dot(weights, np.dot(latest_covariance, weights))
portfolio_std = np.sqrt(portfolio_variance)
VaR = position * z_score * portfolio_std
mVaR = z_score * (np.dot(latest_covariance, weights)) / portfolio_std
cVaR = weights * mVaR
pct_contribution = cVaR / np.sum(cVaR)

component_df = pd.DataFrame({
    'Weight': weights,
    'Marginal_VaR': mVaR,
    'Component_VaR': cVaR,
    'Pct_Contribution': [f'{x:.2%}' for x in pct_contribution]
}, index=stocklist)

print(VaR)
print(component_df)