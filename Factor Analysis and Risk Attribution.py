import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf

stocklist = ["AAPL", "TSLA", "META", "AMZN", "NVDA","MSFT", "PLTR", "GOOGL", "WBD", "F"]

def download_stock_data(tickers, period="5y"):
    data = yf.download(tickers, period=period, auto_adjust=True, group_by='ticker')
    closes = pd.DataFrame()
    for ticker in tickers:
        if ticker in data:
            closes[ticker] = data[ticker]['Close']
    return closes.pct_change().dropna()

returns_data = download_stock_data(stocklist)

def get_fama_french():
    url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"
    ff = pd.read_csv(url, compression='zip', skiprows=3)

    # Clean footer
    ff = ff[~ff.iloc[:, 0].astype(str).str.contains("Copyright", na=False)]

    # Convert dates
    ff['Date'] = pd.to_datetime(ff.iloc[:, 0], format='%Y%m%d', errors='coerce')
    ff = ff.dropna(subset=['Date']).set_index('Date')

    # Convert to decimals
    factors = ff.iloc[:, 1:6].apply(pd.to_numeric, errors='coerce') / 100
    rf = ff.iloc[:, -1].astype(float) / 100

    return factors, rf

factors, rf = get_fama_french()

portfolio_returns = returns_data.mean(axis=1)

# Align dates
common_dates = factors.index.intersection(portfolio_returns.index)
factors = factors.loc[common_dates]
portfolio_returns = portfolio_returns.loc[common_dates]
rf = rf.loc[common_dates]

# Excess returns calculation
portfolio_excess = portfolio_returns - rf.values

X = sm.add_constant(factors)  # Factors already excess returns
model = sm.OLS(portfolio_excess, X).fit()

# Use modern matplotlib style
plt.style.use('default')
sns.set_style("whitegrid")

# Factor Loadings Plot
fig, ax = plt.subplots(figsize=(10, 5))
model.params.drop('const').plot.bar(ax=ax, color='steelblue')
ax.axhline(0, color='black', linestyle='--')
ax.set_title("Fama-French 5-Factor Loadings", pad=20)
ax.set_ylabel("Beta Coefficient")
plt.tight_layout()
plt.savefig('factor_loadings.png', dpi=300)

# Rolling Betas (252-day window)
rolling_betas = pd.DataFrame(index=factors.index, columns=factors.columns)
window = 252

for i in range(window, len(factors)):
    y = portfolio_excess.iloc[i - window:i]
    X_roll = sm.add_constant(factors.iloc[i - window:i])
    roll_model = sm.OLS(y, X_roll).fit()
    rolling_betas.iloc[i] = roll_model.params[1:]

# Plot rolling betas
plt.figure(figsize=(12, 6))
for factor in rolling_betas.columns:
    plt.plot(rolling_betas.index, rolling_betas[factor], label=factor, alpha=0.8)

plt.title("1-Year Rolling Factor Exposures", pad=20)
plt.legend(bbox_to_anchor=(1.05, 1))
plt.tight_layout()
plt.savefig('rolling_betas.png', dpi=300)

print("=" * 80)
print("FAMA-FRENCH 5-FACTOR ANALYSIS RESULTS")
print("=" * 80)
print(f"\nPeriod: {common_dates.min().date()} to {common_dates.max().date()}")
print(f"Number of Observations: {len(common_dates)}")
print(f"\nPortfolio Stocks ({len(stocklist)}): {', '.join(stocklist)}")

print("\n" + model.summary().as_text())

print("\nKEY INTERPRETATION:")
print(f"R-squared: {model.rsquared:.2%} of portfolio variance explained")
print("\nSignificant Factors (p < 0.05):")
sig_factors = model.pvalues[model.pvalues < 0.05].index.tolist()
print(', '.join(sig_factors) if sig_factors else "None")

recommendations = []

if 'Mkt-RF' in sig_factors:
    beta = model.params['Mkt-RF']
    action = "Reduce" if beta > 1 else "Increase" if beta < 0.8 else "Maintain"
    rec = {
        'factor': 'Market',
        'interpretation': f"Beta = {beta:.2f} ({'Aggressive' if beta>1 else 'Defensive'})",
        'action': f"{action} market exposure",
        'tools': ["SPY futures", "SSO (2x ETF)" if beta>1 else "SH (Inverse ETF)"]
    }
    recommendations.append(rec)

if 'SMB' in sig_factors:
    size_effect = model.params['SMB']
    rec = {
        'factor': 'Size (SMB)',
        'interpretation': f"{'Small-cap' if size_effect>0 else 'Large-cap'} tilt",
        'action': "Consider IWM (Russell 2000)" if size_effect>0 else "Shift to IVV (S&P 500)",
        'tools': ["IWM", "VBK" if size_effect>0 else "IVV", "MGC"]
    }
    recommendations.append(rec)

if 'HML' in sig_factors:
    value_effect = model.params['HML']
    rec = {
        'factor': 'Value (HML)',
        'interpretation': f"{'Value' if value_effect>0 else 'Growth'} orientation",
        'action': "Add VTV" if value_effect>0 else "Increase VUG allocation",
        'tools': ["VTV", "RPV" if value_effect>0 else "VUG", "IVW"]
    }
    recommendations.append(rec)

if 'RMW' in sig_factors:
    profitability = model.params['RMW']
    rec = {
        'factor': 'Profitability (RMW)',
        'interpretation': f"{'High profitability' if profitability>0 else 'Low profitability'} exposure",
        'action': "Add QUAL ETF" if profitability>0 else "Screen for ROA improvement",
        'tools': ["QUAL", "SPHQ" if profitability>0 else "FNDX"]
    }
    recommendations.append(rec)

if 'CMA' in sig_factors:
    investment = model.params['CMA']
    rec = {
        'factor': 'Investment (CMA)',
        'interpretation': f"{'Conservative' if investment>0 else 'Aggressive'} investment",
        'action': "Add low-REIT exposure" if investment>0 else "Consider growth REITs",
        'tools': ["VLUE" if investment>0 else "SRET", "REZ"]
    }
    recommendations.append(rec)

# Convert to DataFrame for pretty display
rec_df = pd.DataFrame(recommendations).set_index('factor')
print("\n" + "="*80)
print("FACTOR-SPECIFIC RECOMMENDATIONS")
print("="*80)
print(rec_df[['interpretation', 'action', 'tools']])