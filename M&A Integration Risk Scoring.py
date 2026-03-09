import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Announcement dates of Mastercard's major acquisitions
acquisitions = {
    'Nets':          '2018-08-06',
    'Transfast':     '2019-06-18',
    'Finicity':      '2020-06-23',
    'Dynamic Yield': '2022-03-30',
}

ma  = yf.Ticker("MA")
spy = yf.Ticker("SPY")

# Abnormal return = MA return - SPY return over event window
# Event window: -1 to +5 days around announcement (captures initial reaction)

results = []

for name, date in acquisitions.items():
    announcement = pd.Timestamp(date)
    start        = announcement - pd.offsets.BDay(2)   # 2 days before
    end          = announcement + pd.offsets.BDay(6)   # 6 days after

    ma_price  = ma.history(start=start,  end=end)['Close']
    spy_price = spy.history(start=start, end=end)['Close']

    if len(ma_price) < 3 or len(spy_price) < 3:
        print(f"Insufficient data for {name}, skipping")
        continue

    # Align on same dates
    combined    = pd.DataFrame({'MA': ma_price, 'SPY': spy_price}).dropna()
    ma_return   = (combined['MA'].iloc[-1]  - combined['MA'].iloc[0])  / combined['MA'].iloc[0]
    spy_return  = (combined['SPY'].iloc[-1] - combined['SPY'].iloc[0]) / combined['SPY'].iloc[0]
    abnormal_r  = ma_return - spy_return

    results.append({
        'Acquisition':    name,
        'Date':           date,
        'MA Return':      ma_return,
        'SPY Return':     spy_return,
        'Abnormal Return': abnormal_r
    })

    print(f"{name} ({date})")
    print(f"  MA Return:        {ma_return:.4f}")
    print(f"  SPY Return:       {spy_return:.4f}")
    print(f"  Abnormal Return:  {abnormal_r:.4f}")
    print()

results_df = pd.DataFrame(results)

mean_abnormal   = results_df['Abnormal Return'].mean()
std_abnormal    = results_df['Abnormal Return'].std()
positive_count  = (results_df['Abnormal Return'] > 0).sum()
confidence_pct  = (positive_count / len(results_df)) * 100

print("=" * 60)
print("INTEGRATION CONFIDENCE SCORE")
print("=" * 60)
print(f"Mean Abnormal Return:       {mean_abnormal:.4f}")
print(f"Std Abnormal Return:        {std_abnormal:.4f}")
print(f"Deals with positive AR:     {positive_count} / {len(results_df)}")
print(f"Market Confidence Score:    {confidence_pct:.1f}%")
print()

# Interpret the score
if confidence_pct >= 75:
    print("Interpretation: Market has historically reacted POSITIVELY to Mastercard acquisitions")
    print("                — supports integration risk assumption in synergy model")
elif confidence_pct >= 50:
    print("Interpretation: Market reaction MIXED — moderate integration confidence")
    print("                — apply a haircut to synergy capture rate in Monte Carlo")
else:
    print("Interpretation: Market has historically reacted NEGATIVELY to Mastercard acquisitions")
    print("                — integration risk is elevated, stress test synergy assumptions")

# Once Mastercard announces the Payoneer deal, plug in the date below
# payo_announcement = '2026-XX-XX'
# Add to acquisitions dict and rerun to see where it sits vs historical average

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Abnormal Returns by Deal
colors = ['green' if x > 0 else 'red' for x in results_df['Abnormal Return']]
axes[0].bar(results_df['Acquisition'], results_df['Abnormal Return'], color=colors, alpha=0.7)
axes[0].axhline(mean_abnormal, color='white', linestyle='--', linewidth=1.5,
                label=f'Mean AR: {mean_abnormal:.4f}')
axes[0].axhline(0, color='grey', linestyle='-', linewidth=0.8)
axes[0].set_title('Mastercard Abnormal Returns\nAround Acquisition Announcements')
axes[0].set_xlabel('Acquisition')
axes[0].set_ylabel('Abnormal Return')
axes[0].legend()

# Plot 2: MA vs SPY return comparison
x      = np.arange(len(results_df))
width  = 0.35
axes[1].bar(x - width/2, results_df['MA Return'],  width, label='MA Return',  color='blue',  alpha=0.7)
axes[1].bar(x + width/2, results_df['SPY Return'], width, label='SPY Return', color='orange', alpha=0.7)
axes[1].axhline(0, color='grey', linestyle='-', linewidth=0.8)
axes[1].set_title('MA vs Market Return\nAround Acquisition Announcements')
axes[1].set_xlabel('Acquisition')
axes[1].set_ylabel('Return')
axes[1].set_xticks(x)
axes[1].set_xticklabels(results_df['Acquisition'])
axes[1].legend()

plt.tight_layout()
plt.show()
