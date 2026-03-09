import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ticker = "AAPL"
df = yf.download(tickers=ticker, period="1y", auto_adjust=True)
close = df['Close']
log_returns = np.log(close) - np.log(close.shift(1))
risk_free_rate = 0.05
daily_expected_log_returns = log_returns.mean()
volatility = log_returns.std(ddof=1)

sharpe = (daily_expected_log_returns*252 - risk_free_rate)/(volatility*np.sqrt(252))
print(sharpe)

x= 'AAPL'
data = yf.Ticker(x)
NIFTY = yf.download(x, '2020-01-01', '2024-01-01')
returns = np.log(NIFTY['Close'] / NIFTY['Close'].shift(1)).dropna() #calculate daily log returns
rf_annual = 0.05 #annual risk-free rate
rf_daily = rf_annual / 252 #convert to daily
window = 30 #rolling window
rolling_sharpe = (returns.rolling(window).mean() - rf_daily) /returns.rolling(window).std() #daily rolling Sharpe ratio
plt.plot(rolling_sharpe, label=f'{window}-Day Rolling Sharpe Ratio') #plottingthe rolling Sharpe ratio
plt.title(x + ' Daily Rolling Sharpe Ratio')
plt.xlabel('Date')
plt.ylabel('Sharpe Ratio')
plt.legend()
plt.show() #display graph