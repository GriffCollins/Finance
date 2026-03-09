import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from io import BytesIO
import urllib.request
import zipfile

stocklist = ["AAPL", "TSLA", "META", "AMZN", "NVDA", "MSFT", "PLTR", "GOOGL", "WBD", "F"]

def download_stock_data(tickers, period="5y"):
    data = yf.download(tickers, period=period, auto_adjust=True,
                       group_by='ticker', threads=True, progress=False)
    closes = pd.DataFrame({t: data[t]['Close'] for t in tickers if t in data})
    return closes.pct_change().dropna()

returns_data = download_stock_data(stocklist)

def get_fama_french_6():
    # 5 base factors
    url5 = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"
    # Momentum factor (UMD)
    url_mom = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_daily_CSV.zip"

    def fetch_ff(url, skip=3):
        with urllib.request.urlopen(url) as resp:
            raw = resp.read()
        with zipfile.ZipFile(BytesIO(raw)) as z:
            fname = z.namelist()[0]
            with z.open(fname) as f:
                df = pd.read_csv(f, skiprows=skip, header=0)
        # Drop copyright footer rows
        df = df[pd.to_numeric(df.iloc[:, 0], errors='coerce').notna()]
        df['Date'] = pd.to_datetime(df.iloc[:, 0].astype(int).astype(str), format='%Y%m%d')
        df = df.set_index('Date')
        return df.iloc[:, 1:].apply(pd.to_numeric, errors='coerce') / 100

    ff5 = fetch_ff(url5)
    ff5.columns = ff5.columns.str.strip()

    mom = fetch_ff(url_mom, skip=13)
    mom.columns = ['UMD']

    ff6 = ff5.join(mom, how='inner')
    rf  = ff6.pop('RF')          # separate out risk-free rate
    return ff6, rf

factors, rf = get_fama_french_6()

portfolio_returns = returns_data.mean(axis=1)
common_dates = factors.index.intersection(portfolio_returns.index)
factors           = factors.loc[common_dates]
portfolio_returns = portfolio_returns.loc[common_dates]
rf                = rf.loc[common_dates]
portfolio_excess  = portfolio_returns - rf.values

X = sm.add_constant(factors)
model = sm.OLS(portfolio_excess, X).fit()

window = 252
factor_cols = factors.columns.tolist()
n_factors   = len(factor_cols)

y_arr = portfolio_excess.values
X_arr = np.column_stack([np.ones(len(factors)), factors.values])  # (T, 1+6)

rolling_betas = np.full((len(factors), n_factors), np.nan)

for i in range(window, len(factors)):
    y_w = y_arr[i - window:i]
    X_w = X_arr[i - window:i]
    coeffs, *_ = np.linalg.lstsq(X_w, y_w, rcond=None)
    rolling_betas[i] = coeffs[1:]   # drop intercept

rolling_betas_df = pd.DataFrame(rolling_betas, index=factors.index, columns=factor_cols)

plt.style.use('default')
sns.set_style("whitegrid")

fig, ax = plt.subplots(figsize=(10, 5))
model.params.drop('const').plot.bar(ax=ax, color='steelblue')
ax.axhline(0, color='black', linestyle='--')
ax.set_title("Fama-French 6-Factor Loadings (incl. Momentum)", pad=20)
ax.set_ylabel("Beta Coefficient")
plt.tight_layout()
plt.savefig('factor_loadings.png', dpi=300)

plt.figure(figsize=(12, 6))
for factor in factor_cols:
    plt.plot(rolling_betas_df.index, rolling_betas_df[factor], label=factor, alpha=0.8)
plt.title("1-Year Rolling Factor Exposures (6-Factor)", pad=20)
plt.legend(bbox_to_anchor=(1.05, 1))
plt.tight_layout()
plt.savefig('rolling_betas.png', dpi=300)

print("=" * 80)
print("FAMA-FRENCH 6-FACTOR ANALYSIS RESULTS")
print("=" * 80)
print(f"\nPeriod: {common_dates.min().date()} to {common_dates.max().date()}")
print(f"Observations: {len(common_dates)}")
print(f"Portfolio ({len(stocklist)} stocks): {', '.join(stocklist)}")
print("\n" + model.summary().as_text())

print(f"\nR-squared: {model.rsquared:.2%} of portfolio variance explained")
sig_factors = model.pvalues[model.pvalues < 0.05].index.tolist()
print("Significant Factors (p < 0.05):", ', '.join(sig_factors) if sig_factors else "None")

RECS = {
    'Mkt-RF': lambda b: {
        'interpretation': f"Market Beta = {b:.2f} ({'Aggressive' if b>1 else 'Defensive'})",
        'action': ('Reduce' if b>1 else 'Increase' if b<0.8 else 'Maintain') + ' market exposure',
        'tools': ["SPY futures", "SSO" if b>1 else "SH"]
    },
    'SMB': lambda b: {
        'interpretation': f"{'Small' if b>0 else 'Large'}-cap tilt",
        'action': "Add IWM (Russell 2000)" if b>0 else "Shift to IVV (S&P 500)",
        'tools': ["IWM", "VBK"] if b>0 else ["IVV", "MGC"]
    },
    'HML': lambda b: {
        'interpretation': f"{'Value' if b>0 else 'Growth'} orientation",
        'action': "Add VTV" if b>0 else "Increase VUG allocation",
        'tools': ["VTV", "RPV"] if b>0 else ["VUG", "IVW"]
    },
    'RMW': lambda b: {
        'interpretation': f"{'High' if b>0 else 'Low'} profitability exposure",
        'action': "Add QUAL ETF" if b>0 else "Screen for ROA improvement",
        'tools': ["QUAL", "SPHQ"] if b>0 else ["FNDX"]
    },
    'CMA': lambda b: {
        'interpretation': f"{'Conservative' if b>0 else 'Aggressive'} investment",
        'action': "Low-REIT exposure" if b>0 else "Consider growth REITs",
        'tools': ["VLUE"] if b>0 else ["SRET", "REZ"]
    },
    'UMD': lambda b: {
        'interpretation': f"{'Positive' if b>0 else 'Negative'} momentum tilt",
        'action': "Add MTUM ETF" if b>0 else "Reduce momentum exposure / add mean-reversion",
        'tools': ["MTUM", "QMOM"] if b>0 else ["DWAS"]
    },
}

recommendations = []
for factor in factor_cols:
    if factor in sig_factors and factor in RECS:
        rec = RECS[factor](model.params[factor])
        rec['factor'] = factor
        recommendations.append(rec)

if recommendations:
    rec_df = pd.DataFrame(recommendations).set_index('factor')
    print("\n" + "="*80)
    print("FACTOR-SPECIFIC RECOMMENDATIONS")
    print("="*80)
    print(rec_df[['interpretation', 'action', 'tools']])