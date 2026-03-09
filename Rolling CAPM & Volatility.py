import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from datetime import datetime
import yfinance as yf

stock = ["PLTR"]
market = ["SPY"]
raw_spy_data = yf.download(market, auto_adjust=True, period="max")
raw_stock_data = yf.download(stock, auto_adjust=True, period="max")
print(raw_stock_data.head())
print(raw_spy_data.head())
returns_data = raw_stock_data["Close"].pct_change().dropna()
spy_data = raw_spy_data["Close"].pct_change().dropna()

start = datetime(2023, 1, 1)
end = datetime(2025, 7, 14)

stockname  = "PLTR"

#Create the stock and market dataframes then combine them to be used in the statsmodels function
stock_and_market = pd.DataFrame({
    'stock': returns_data[stockname],
    'market': spy_data["SPY"]
}).dropna()

#Run linear regression to find beta using a 60-day rolling window
def rolling_beta(df, window):
    #Initialise the dataframe 
    results = pd.DataFrame(index=df.index, columns=["alpha", "beta"])

    for i in range(window, len(df)):
        y = df["stock"].iloc[i-window:i]  # Stock returns
        X = sm.add_constant(df['market'].iloc[i-window:i])  # Market returns
        
        model = sm.OLS(y, X).fit()
        results.iloc[i] = [model.params['const'], model.params['market']]
    
    return results.dropna()

#Rolling volatility function and multiply by sqrt(252) to annualise
def rolling_volatility(df, window, trading_days):
    return df.rolling(window).std() * np.sqrt(trading_days)

#Create dataframe
rolling_params = rolling_beta(stock_and_market, 60)

#Alter format
results = pd.concat([stock_and_market, rolling_params], axis=1).dropna()

#Add rolling volatility
results['Stock_Volatility'] = rolling_volatility(results['stock'], 60, 252)
results['Market_Volatility'] = rolling_volatility(results['market'], 60, 252)

#View
print(results.tail())

#Visualise using matplotlib

plt.figure(figsize=(14, 7))
plt.subplot(2, 1, 1)
results['beta'].plot(title='60-Day Rolling Beta (' + stockname +' vs SPY)')
plt.axhline(1, color='r', linestyle='--')
plt.ylabel('Beta')
plt.grid()

plt.subplot(2, 1, 2)
results['alpha'].plot(title='60-Day Rolling Alpha')
plt.axhline(0, color='r', linestyle='--')
plt.ylabel('Alpha')
plt.grid()

plt.tight_layout()
plt.show()

plt.figure(figsize=(14, 5))
results['Stock_Volatility'].plot(label=""+stockname+" Volatility", color='blue')
results['Market_Volatility'].plot(label='SPY Volatility', color='orange', alpha=0.7)
plt.title('60-Day Rolling Annualized Volatility')
plt.ylabel('Volatility')
plt.legend()
plt.grid()
plt.show()










