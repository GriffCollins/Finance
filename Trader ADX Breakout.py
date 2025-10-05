import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""
ADX Breakout Trader
Long-only when ADX turns up when below both positive and negative directional lines
And when positive directional line is above negative directional line
Sell when ADX turn back down
ADX alone isn't a good strategy, more parameters are required
"""

# Initial parameters
ticker = 'SPY'
length = 20
commission = 0   # Assuming 0% commission online broker
slippage = 0.0005     # 0.05% round-trip slippage reasonable for no limits on highly liquid stocks
initial_capital = 100000.0
bars_per_day = 1
risk_free_rate = 0.05
trade_cost = commission + slippage

#Retrieve stock data
df = yf.download(ticker, period='max', progress=False, auto_adjust=True, interval='1d', multi_level_index=False)
df = df[['Close', 'High', 'Low']].dropna()
df['signal'] = 0

#Signal precalculations
df['UpMove'] = df['High'] - df['High'].shift(1)
df['DownMove'] = df['Low'].shift(1) - df['Low']
df['+DM'] = 0
df['-DM'] = 0
df.loc[(df['UpMove'] > df['DownMove']) & (df['UpMove'] > 0), '+DM'] = df['UpMove']
df.loc[(df['UpMove'] < df['DownMove']) & (df['DownMove'] > 0), '-DM'] = df['DownMove']
df['TrueRange'] = pd.concat([df['High'] - df['Low'],(df['High'] - df['Close'].shift(1)).abs(),(df['Low'] - df['Close'].shift(1)).abs()], axis=1).max(axis=1)
df['+DI'] = 100 * df['+DM'].ewm(alpha=1/length, adjust=False).mean() / df['TrueRange'].ewm(alpha=1/length, adjust=False).mean()
df['-DI'] = 100 * df['-DM'].ewm(alpha=1/length, adjust=False).mean() / df['TrueRange'].ewm(alpha=1/length, adjust=False).mean()
df['ADX'] = 100 * (abs(df['+DI'] - df['-DI']) / (df['+DI'] + df['-DI'])).ewm(alpha=1/length, adjust=False).mean()

#Signal generator
def signal(df):
    adx = df["ADX"]
    plusDI = df["+DI"]
    negDI = df["-DI"]
    adx_prev = adx.shift(1)

    signal = []
    state = 0

    for i in range(len(df)):
        if i == 0:
            signal.append(0)
            continue

        H = adx.iloc[i] > adx_prev.iloc[i]
        A = (
            (adx.iloc[i] < plusDI.iloc[i]) and
            (adx.iloc[i] < negDI.iloc[i]) and
            H and
            (plusDI.iloc[i] > negDI.iloc[i])
        )

        if state == 0 and A:
            state = 1
        elif state == 1 and not H:
            state = 0

        signal.append(state)

    return pd.Series(signal, index=df.index)

# Calculations
df['signal'] = signal(df)
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
plt.plot(df.index, df['ADX'], label='Net Equity', linewidth=1)
plt.ylabel('Average Directional Index')
plt.xlabel('Date')
plt.legend()
plt.title(f'{ticker} Average Directional Index')
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
plt.title(f'{ticker} Moving Average Crossover Strategy')
plt.grid(True, alpha=0.3)
plt.show()

#Create dataframe csv
df.to_csv('ADX Breakout Trader.csv')