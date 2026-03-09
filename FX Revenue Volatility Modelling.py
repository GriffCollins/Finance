import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# M&A results for Payoneer

payo          = yf.Ticker("PAYO")
financials    = payo.financials
info          = payo.info
annual_revenue = financials.loc['Total Revenue'].iloc[0]
print("=" * 60)
print("PAYONEER DATA")
print("=" * 60)
print(f"Annual Revenue:  ${annual_revenue:,.0f}")

currency_pairs = {
    'EUR/USD': 'EURUSD=X',
    'GBP/USD': 'GBPUSD=X',
    'CNY/USD': 'CNY=X',
    'BRL/USD': 'BRLUSD=X',
    'INR/USD': 'INRUSD=X',
    'JPY/USD': 'JPYUSD=X',
}

revenue_weights = {
    'EUR/USD': 0.25,
    'GBP/USD': 0.15,
    'CNY/USD': 0.20,
    'BRL/USD': 0.15,
    'INR/USD': 0.15,
    'JPY/USD': 0.10,
}

print("\n" + "=" * 60)
print("FX DATA PULL")
print("=" * 60)

fx_data = {}
for name, ticker in currency_pairs.items():
    raw = yf.download(ticker, start='2020-01-01', auto_adjust=True, progress=False)['Close'].squeeze()
    fx_data[name] = raw.dropna()
    print(f"{name}: {len(fx_data[name])} observations")

fx_df      = pd.DataFrame(fx_data).dropna()
fx_returns = fx_df.pct_change().dropna()

print("\n" + "=" * 60)
print("ANNUALISED FX VOLATILITY")
print("=" * 60)

fx_vol = fx_returns.std() * np.sqrt(252)
for pair, vol in fx_vol.items():
    if vol > 0.15:
        verdict = "HIGH — significant inherited risk"
    elif vol > 0.08:
        verdict = "MODERATE — manageable with hedging"
    else:
        verdict = "LOW — minimal risk"
    print(f"{pair}: {vol:.4f}  →  {verdict}")

confidence    = 0.95
daily_revenue = annual_revenue / 252

print("\n" + "=" * 60)
print(f"VALUE AT RISK ({confidence*100:.0f}% CONFIDENCE) — DAILY REVENUE IMPACT")
print("=" * 60)

var_records = []
for pair in fx_returns.columns:
    weight      = revenue_weights[pair]
    exposed_rev = daily_revenue * weight
    var_pct     = np.percentile(fx_returns[pair], (1 - confidence) * 100)
    var_dollar  = abs(var_pct * exposed_rev)
    var_records.append({
        'Pair':            pair,
        'Weight':          weight,
        'Exposed Revenue': exposed_rev,
        'VaR %':           var_pct,
        'VaR $':           var_dollar
    })
    print(f"{pair}: ${var_dollar:,.0f} daily at risk ({var_pct:.4f})")

var_df           = pd.DataFrame(var_records)
total_var_daily  = var_df['VaR $'].sum()
total_var_annual = total_var_daily * 252
var_pct_revenue  = (total_var_annual / annual_revenue) * 100

print(f"\nTotal Portfolio VaR (daily):    ${total_var_daily:,.0f}")
print(f"Total Portfolio VaR (annual):   ${total_var_annual:,.0f}")
print(f"As % of annual revenue:         {var_pct_revenue:.2f}%")

if var_pct_revenue < 5:
    print("Verdict: FX risk is LOW relative to revenue — deal thesis intact")
elif var_pct_revenue < 15:
    print("Verdict: FX risk is MODERATE — Mastercard hedging infrastructure mitigates this")
else:
    print("Verdict: FX risk is HIGH — stress test synergy assumptions accordingly")

num_scenarios  = 100000
fx_shock_means = fx_returns.mean().values
fx_shock_stds  = fx_returns.std().values
weights        = np.array(list(revenue_weights.values()))

fx_shocks      = np.random.normal(fx_shock_means, fx_shock_stds,
                                   size=(num_scenarios, len(currency_pairs)))
revenue_impacts = fx_shocks * weights * annual_revenue
total_impacts   = revenue_impacts.sum(axis=1) * 252

mean_impact = np.mean(total_impacts)
var_95      = np.percentile(total_impacts, 5)
var_99      = np.percentile(total_impacts, 1)
synergies   = 3_346_000_000

print("\n" + "=" * 60)
print("MONTE CARLO FX REVENUE STRESS TEST")
print("=" * 60)
print(f"Mean Annual FX Impact:          ${mean_impact:,.0f}")
print(f"VaR 95% (annual):               ${var_95:,.0f}")
print(f"VaR 99% (annual):               ${var_99:,.0f}")
print(f"VaR 95% as % of revenue:        {(var_95/annual_revenue)*100:.2f}%")
print(f"VaR 99% as % of revenue:        {(var_99/annual_revenue)*100:.2f}%")
print(f"\nSynergies:                      ${synergies:,.0f}")
print(f"VaR 95% as % of synergies:      {(abs(var_95)/synergies)*100:.2f}%")

if abs(var_95) < synergies * 0.1:
    print("Verdict: FX downside is less than 10% of synergies — deal economics robust")
elif abs(var_95) < synergies * 0.25:
    print("Verdict: FX downside is material but synergies still dominate")
else:
    print("Verdict: FX risk materially threatens synergy case — flag on risk slide")

corr_matrix = fx_returns.corr()

print("\n" + "=" * 60)
print("FX CORRELATION MATRIX")
print("=" * 60)
print(corr_matrix.round(3))

high_corr_pairs = [(corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.values[i,j])
                   for i in range(len(corr_matrix))
                   for j in range(i+1, len(corr_matrix))
                   if abs(corr_matrix.values[i,j]) > 0.7]

if high_corr_pairs:
    print("\nHighly correlated pairs (>0.7) — risk compounds, not diversified:")
    for p1, p2, r in high_corr_pairs:
        print(f"  {p1} / {p2}: {r:.3f}")
else:
    print("\nNo highly correlated pairs — FX risk diversifies across regions")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("Payoneer FX / Revenue Volatility Analysis", fontsize=14)

# Plot 1: Annualised Volatility
axes[0, 0].bar(fx_vol.index, fx_vol.values, color='steelblue', alpha=0.7)
axes[0, 0].set_title('Annualised FX Volatility by Currency Pair')
axes[0, 0].set_ylabel('Volatility')
axes[0, 0].set_xlabel('Currency Pair')

# Plot 2: VaR by Currency
axes[0, 1].bar(var_df['Pair'], var_df['VaR $'], color='crimson', alpha=0.7)
axes[0, 1].set_title(f'Daily VaR by Currency ({confidence*100:.0f}% Confidence)')
axes[0, 1].set_ylabel('Daily Revenue at Risk ($)')
axes[0, 1].set_xlabel('Currency Pair')

# Plot 3: Monte Carlo Distribution
axes[1, 0].hist(total_impacts, bins=50, alpha=0.5, color='blue')
axes[1, 0].axvline(mean_impact, color='red',    linestyle='--', label=f'Mean: ${mean_impact:,.0f}')
axes[1, 0].axvline(var_95,      color='orange', linestyle='--', label=f'VaR 95%: ${var_95:,.0f}')
axes[1, 0].axvline(var_99,      color='white',  linestyle='--', label=f'VaR 99%: ${var_99:,.0f}')
axes[1, 0].axvline(-synergies,  color='green',  linestyle='-',  linewidth=2,
                   label=f'Synergies: ${synergies/1e9:.1f}B')
axes[1, 0].set_title('Monte Carlo Annual FX Revenue Impact vs Synergies')
axes[1, 0].set_xlabel('Annual Revenue Impact ($)')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].legend()

# Plot 4: Correlation Heatmap
im = axes[1, 1].imshow(corr_matrix.values, cmap='coolwarm', vmin=-1, vmax=1)
axes[1, 1].set_xticks(range(len(corr_matrix.columns)))
axes[1, 1].set_yticks(range(len(corr_matrix.columns)))
axes[1, 1].set_xticklabels(corr_matrix.columns, rotation=45)
axes[1, 1].set_yticklabels(corr_matrix.columns)
axes[1, 1].set_title('FX Correlation Matrix')
plt.colorbar(im, ax=axes[1, 1])

for i in range(len(corr_matrix)):
    for j in range(len(corr_matrix.columns)):
        axes[1, 1].text(j, i, f'{corr_matrix.values[i, j]:.2f}',
                        ha='center', va='center', fontsize=8)

plt.tight_layout()
plt.show()