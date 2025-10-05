import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt

"""
Bollinger Bands Trader
There are 2 versions, mean reverting and trending.
Mean reverting: Long when < -2sigma , Short when > +2sigma
Trending: Long when > +2sigma , Short when < -2sigma
"""

# Initial parameters
ticker = 'TSLA'
start = '2025-07-16'
end = '2025-09-8'
length = 20
commission = 0   # Assuming 0% commission online broker
slippage = 0.0005     # 0.05% round-trip slippage reasonable for no limits on highly liquid stocks
initial_capital = 100000.0
bars_per_day = 78
risk_free_rate = 0.05
trade_cost = commission + slippage
reverse = False #Mean reverting = True, Trending = False

#Retrieve stock data and initialise signal column
df = yf.download(ticker, start=start,end=end, progress=False, auto_adjust=True, interval='5m', multi_level_index=False)
df = df[['Close']].dropna().reset_index(drop=True)
df['signal'] = 0

#Signal generator
df['moving_average'] = df['Close'].rolling(length, min_periods=length).mean()
df['moving_std'] = df['Close'].rolling(length, min_periods=length).std(ddof=1)
df['bollingerband_2+'] = df['moving_average'] + 2*df['moving_std']
df['bollingerband_2-'] = df['moving_average'] - 2*df['moving_std']

if reverse:
    df.loc[df['Close'] < df['bollingerband_2-'], 'signal'] = 1
    df.loc[df['Close'] > df['bollingerband_2+'], 'signal'] = -1
else:
    df.loc[df['Close'] < df['bollingerband_2-'], 'signal'] = -1
    df.loc[df['Close'] > df['bollingerband_2+'], 'signal'] = 1

# Calculations
df['position'] = df['signal']
df['position_change'] = df['position'].diff().fillna(0)
df['trade_costs'] = 0.0
df.loc[df['position_change'] != 0, 'trade_costs'] = trade_cost

df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1)).fillna(0)
df['strategy_ret_gross'] = df['position'].shift(1).fillna(0) * df['log_returns']
df['strategy_ret_net'] = df['strategy_ret_gross'] - df['trade_costs']

df['cumlogret_gross'] = df['strategy_ret_gross'].cumsum()
df['cumlogret_net']   = df['strategy_ret_net'].cumsum()
df['cumlogret_buyhold'] = df['log_returns'].cumsum()

df['equity_gross'] = initial_capital * np.exp(df['cumlogret_gross'])
df['equity_net']   = initial_capital * np.exp(df['cumlogret_net'])
df['equity_buyhold'] = initial_capital * np.exp(df['cumlogret_buyhold'])

# Return and drawdown
def annualise_return(cumlogret, days_per_year=252, bars_per_day=bars_per_day):
    if len(cumlogret) == 0:
        return np.nan
    total_bars = len(cumlogret)
    years = total_bars / (days_per_year * bars_per_day)
    final_ret = np.exp(cumlogret[-1]) - 1
    if final_ret <= -1:
        return -1.0
    return (1 + final_ret) ** (1/years) - 1

def max_drawdown(equity_curve):
    if equity_curve.empty:
        return np.nan
    roll_max = equity_curve.cummax()
    drawdown = equity_curve / roll_max - 1.0
    return drawdown.min()

# Performance metrics
mdd_gross = max_drawdown(df['equity_gross'])
ann_ret_gross = annualise_return(df['cumlogret_gross'].values)
ann_vol_gross = df['strategy_ret_gross'].std() * np.sqrt(252 * bars_per_day)
sharpe_gross = (ann_ret_gross - risk_free_rate) / ann_vol_gross if ann_vol_gross != 0 else np.nan

mdd_net = max_drawdown(df['equity_net'])
ann_ret_net = annualise_return(df['cumlogret_net'].values)
ann_vol_net = df['strategy_ret_net'].std() * np.sqrt(252 * bars_per_day)
sharpe_net = (ann_ret_net - risk_free_rate) / ann_vol_net if ann_vol_net != 0 else np.nan

mdd_buyhold = max_drawdown(df['equity_buyhold'])
ann_ret_buyhold = annualise_return(df['cumlogret_buyhold'].values)
ann_vol_buyhold = df['log_returns'].std() * np.sqrt(252 * bars_per_day)
sharpe_buyhold = (ann_ret_buyhold - risk_free_rate) / ann_vol_buyhold if ann_vol_buyhold != 0 else np.nan

total_trades = (df['position_change'] != 0).sum()
win_rate = (df['strategy_ret_net'] > 0).mean() if len(df) > 0 else 0

# Summary
print('Ticker:         ', ticker)
print(f'Initial capital: ${initial_capital:,.0f}')
print(f'Total bars:      {len(df)}')
print(f'Total trades:    {total_trades}')

print('\n--- Gross (before costs) ---')
print(f'Annualised return: {ann_ret_gross:.2%}')
print(f'Annualised vol:   {ann_vol_gross:.2%}')
print(f'Sharpe (rf={risk_free_rate:.2f}):    {sharpe_gross:.2f}')
print(f'Max Drawdown:     {mdd_gross:.2%}')
print(f'Final Gross Equity: ${df['equity_gross'].iloc[-1]:,.2f}')

print('\n--- Net (after costs) ---')
print(f'Annualised return: {ann_ret_net:.2%}')
print(f'Annualised vol:   {ann_vol_net:.2%}')
print(f'Sharpe (rf={risk_free_rate:.2f}):    {sharpe_net:.2f}')
print(f'Max Drawdown:     {mdd_net:.2%}')
print(f'Final Net Equity:   ${df['equity_net'].iloc[-1]:,.2f}')

print('\n--- Buy and Hold ---')
print(f'Annualised Return: {ann_ret_buyhold:.2%}')
print(f'Annualised vol:   {ann_vol_buyhold:.2%}')
print(f'Sharpe (rf={risk_free_rate:.2f}):    {sharpe_buyhold:.2f}')
print(f'Max Drawdown: {mdd_buyhold:.2%}')
print(f'Final Net Equity: ${df['equity_buyhold'].iloc[-1]:,.2f}')

#Plot closing price with bollinger bands
plt.figure(figsize=(24, 6))
plt.plot(df.index, df['moving_average'], label='Smoothed Close', linewidth=1, color='r')
plt.plot(df.index, df['bollingerband_2+'], label='+2 sigma Close', linewidth=1)
plt.plot(df.index, df['bollingerband_2-'], label='-2 sigma Close', linewidth=1, color='g')
plt.ylabel('Close')
plt.xlabel('Candlestick Number')
plt.legend()
plt.title(f'{ticker} Time Series Closing Price with Bollinger Bands')
plt.grid(True, alpha=0.3)
plt.show()

# Plot equity curve
plt.figure(figsize=(12, 6))
plt.step(df.index, df['equity_net'], label='Net Equity', linewidth=1)
plt.step(df.index, df['equity_gross'], label='Gross Equity', linewidth=1, alpha=0.7)
plt.step(df.index, df['equity_buyhold'], label='Buy & Hold', linewidth=1, alpha=0.7)
plt.ylabel('Equity ($)')
plt.xlabel('Date')
plt.legend()
plt.title(f'{ticker} Bollinger Bands Strategy')
plt.grid(True, alpha=0.3)
plt.show()

#Create dataframe csv
df.to_csv('Bollinger Bands Trader.csv')