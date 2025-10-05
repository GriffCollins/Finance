import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""
Stochastic Slow Trader
May generate excessive signals in range markets
Can remain in overbought/oversold during strong trends, yielding false reversals
According to George Lane, the Stochastics indicator is to be used with:
cycles, Elliott Wave Theory and Fibonacci retracement for timing
"""

# Initial parameters
ticker = 'SPY'
length = 3
commission = 0   # Assuming 0% commission online broker
slippage = 0.0005     # 0.05% round-trip slippage reasonable for no limits on highly liquid stocks
initial_capital = 100000.0
bars_per_day = 1
risk_free_rate = 0.05
trade_cost = commission + slippage
reverse = True
multiplier = 3

#Retrieve stock data
df = yf.download(ticker, period='5y', progress=False, auto_adjust=True, interval='1d', multi_level_index=False)
df = df[['Close', 'High', 'Low']].dropna()
df['signal'] = 0

df['TrueRange'] = pd.concat([df['High'] - df['Low'],(df['High'] - df['Close'].shift(1)).abs(),(df['Low'] - df['Close'].shift(1)).abs()], axis=1).max(axis=1)
df['ATR'] = df['TrueRange'].ewm(alpha=1/length, adjust=False).mean()
df['BUB'] = (df['High'] + df['Low']) / 2 + multiplier * df['ATR']
df['BLB'] = (df['High'] + df['Low']) / 2 - multiplier * df['ATR']


def signal(df):
    bub = df['BUB']
    blb = df['BLB']
    close = df['Close']

    # Initialize lists
    fub = [bub.iloc[0]]
    flb = [blb.iloc[0]]
    signal = [0]  # 0 = neutral/bearish, 1 = bullish
    state = 0  # initial trend: 0 = bearish, 1 = bullish

    for i in range(1, len(df)):
        # Final Upper Band
        if bub.iloc[i] < fub[i - 1] or close.iloc[i - 1] > fub[i - 1]:
            fub.append(bub.iloc[i])
        else:
            fub.append(fub[i - 1])

        # Final Lower Band
        if blb.iloc[i] > flb[i - 1] or close.iloc[i - 1] < flb[i - 1]:
            flb.append(blb[i])
        else:
            flb.append(flb[i - 1])

        # Determine trend
        if state == 0 and close.iloc[i] > fub[i]:
            state = 1
        elif state == 1 and close.iloc[i] < flb[i]:
            state = 0

        signal.append(state)

    return pd.Series(signal, index=df.index), pd.Series(fub, index=df.index), pd.Series(flb, index=df.index)


# Calculations
df['signal'], df['FUB'], df['FLB'] = signal(df)
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

#Plot average directional index
plt.figure(figsize=(24, 6))
plt.plot(df.index, df['Close'], label='Close', linewidth=1, color='C0')
plt.plot(df.index, df['FUB'], label='FUB', linewidth=1)
plt.plot(df.index, df['FLB'], label='FLB', linewidth=1)
plt.ylabel('Momentum Indicator')
plt.xlabel('Date')
plt.legend()
plt.title(f'{ticker} Supertrend Close Timeseries')
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
plt.title(f'{ticker} Supertrend Strategy')
plt.grid(True, alpha=0.3)
plt.show()

#Create dataframe csv
df.to_csv('Supertrend Trader.csv')