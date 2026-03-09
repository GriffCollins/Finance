import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import fredapi
import pandas as pd

payo          = yf.Ticker("PAYO")
financials    = payo.financials
balance_sheet = payo.balance_sheet
info          = payo.info

print("=" * 60)
print("PAYONEER DIAGNOSTICS")
print("=" * 60)

revenue = financials.loc['Total Revenue']
ebit    = financials.loc['EBIT']

print("\nREVENUE:")
print(revenue)
print("\nEBIT:")
print(ebit)

growth_rates = revenue.pct_change(periods=-1)
print("\nYOY REVENUE GROWTH:")
print(growth_rates)

latest_revenue = revenue.iloc[0]
latest_ebit    = ebit.iloc[0]
total_debt     = balance_sheet.loc['Total Debt'].iloc[0] if 'Total Debt' in balance_sheet.index else 0
cash           = balance_sheet.loc['Cash And Cash Equivalents'].iloc[0]
market_cap     = info['marketCap']

print(f"\nMarket Cap: {market_cap:,.0f}")
print(f"Total Debt: {total_debt:,.0f}")
print(f"Cash:       {cash:,.0f}")

tax_provision = financials.loc['Tax Provision'].iloc[0]
pretax_income = financials.loc['Pretax Income'].iloc[0]
tax_rate      = max(0.156, tax_provision / pretax_income) if pretax_income > 0 else 0.156
print(f"\nEffective Tax Rate: {tax_rate:.4f}")

FRED_API_KEY = "f022be36d1b0a0ad0437176d04939ad8"
fred         = fredapi.Fred(api_key=FRED_API_KEY)

risk_free      = fred.get_series('DGS10').dropna().iloc[-1] / 100
erp            = 0.055
beta           = info['beta']
cost_of_equity = risk_free + beta * erp
cost_of_debt   = fred.get_series('BAA').dropna().iloc[-1] / 100
weight_equity  = market_cap / (market_cap + total_debt)
weight_debt    = total_debt / (market_cap + total_debt)
WACC           = weight_equity * cost_of_equity + weight_debt * cost_of_debt * (1 - tax_rate)

breakeven_inflation = fred.get_series('T10YIE').dropna().iloc[-1] / 100
gdp_series          = fred.get_series('A191RL1Q225SBEA')
long_run_gdp        = gdp_series.dropna().tail(20).mean() / 100
terminal_growth     = min(0.03, 0.5 * breakeven_inflation + 0.5 * long_run_gdp)

print("\n" + "=" * 60)
print("CAPITAL MARKET INPUTS")
print("=" * 60)
print(f"Risk Free:        {risk_free:.4f}")
print(f"ERP:              {erp:.4f}")
print(f"Beta:             {beta:.4f}")
print(f"Cost of Equity:   {cost_of_equity:.4f}")
print(f"Cost of Debt:     {cost_of_debt:.4f}")
print(f"WACC:             {WACC:.4f}")
print(f"Terminal Growth:  {terminal_growth:.4f}")

num_scenarios  = 100000
T              = 8
g_mean         = growth_rates.mean()
g_vol          = growth_rates.std()
phi            = 0.5          # mean reversion speed for growth
margin_base    = latest_ebit / latest_revenue
margin_target  = 0.20         # long-run fintech EBIT margin
margin_vol     = 0.02
sales_to_capital = 3.0

# ROIC parameters — used in sensitivity, derived here consistently
ROIC_MEAN = sales_to_capital * margin_target * (1 - tax_rate)
ROIC_STD  = 0.03

enterprise_values = []

for _ in range(num_scenarios):
    revenue_path = [latest_revenue]
    margin_path  = [margin_base]
    growth_prev  = np.random.normal(g_mean, g_vol)

    for t in range(T):
        growth      = g_mean + phi * (growth_prev - g_mean) + np.random.normal(0, g_vol)
        growth      = max(growth, -0.8)
        growth_prev = growth
        revenue_path.append(revenue_path[-1] * (1 + growth))

        margin = margin_path[-1] + 0.35 * (margin_target - margin_path[-1]) \
                 + np.random.normal(0, margin_vol)
        margin_path.append(max(margin, -0.2))

    fcff = []
    for t in range(1, len(revenue_path)):
        delta_rev   = revenue_path[t] - revenue_path[t - 1]
        nopat       = revenue_path[t] * margin_path[t] * (1 - tax_rate)
        reinvestment = max(delta_rev / sales_to_capital, 0)
        fcff.append(nopat - reinvestment)

    pv = sum(cf / (1 + WACC) ** (t + 1) for t, cf in enumerate(fcff))

    if WACC > terminal_growth:
        tv  = (fcff[-1] * (1 + terminal_growth)) / (WACC - terminal_growth)
        pv += tv / (1 + WACC) ** T

    enterprise_values.append(pv)

enterprise_values = np.array(enterprise_values)

deal_price   = 4_372_800_000
deal_price   = 3_386_180_000
mean_ev      = enterprise_values.mean()
ci_low, ci_high = np.percentile(enterprise_values, [2.5, 97.5])
prob_payoff  = np.mean(enterprise_values > deal_price) * 100
market_ev    = market_cap + total_debt - cash  # defined here for sensitivity use

print("\n" + "=" * 60)
print("VALUATION RESULTS")
print("=" * 60)
print(f"Mean Enterprise Value:     {mean_ev:,.0f}")
print(f"95% CI:                    [{ci_low:,.0f}, {ci_high:,.0f}]")
print(f"Deal Price:                {deal_price:,.0f}")
print(f"Probability Deal Pays Off: {prob_payoff:.2f}%")

plt.figure(figsize=(12, 8))
plt.hist(enterprise_values, bins=50, alpha=0.6, color='blue')
plt.axvline(mean_ev,    color='red',    linestyle='--', label=f"Mean: {mean_ev:,.0f}")
plt.axvline(ci_low,     color='green',  linestyle='--', label=f"95% CI Lower: {ci_low:,.0f}")
plt.axvline(ci_high,    color='orange', linestyle='--', label=f"95% CI Upper: {ci_high:,.0f}")
plt.axvline(deal_price, color='white',  linewidth=2,
            label=f"Deal Price: ${deal_price/1e9:.2f}B | Payoff: {prob_payoff:.1f}%")
plt.title("Payoneer Monte Carlo DCF — FCFF Model")
plt.xlabel("Enterprise Value ($)")
plt.ylabel("Frequency")
plt.legend()
plt.tight_layout()
plt.show()

margin_range = np.linspace(0.10, 0.35, 15)
wacc_range   = np.linspace(WACC - 0.02, WACC + 0.02, 15)

sensitivity_matrix = np.zeros((len(margin_range), len(wacc_range)))

print("\n" + "=" * 60)
print("SENSITIVITY ANALYSIS — running...")
print("=" * 60)

for i, terminal_margin in enumerate(margin_range):
    for j, wacc_test in enumerate(wacc_range):

        ev_temp = []

        for _ in range(3000):
            rev         = latest_revenue
            fcf_stream  = []
            growth      = np.random.normal(g_mean, g_vol)
            roic        = max(0.05, np.random.normal(ROIC_MEAN, ROIC_STD))

            for year in range(T):
                rev    *= (1 + growth)
                margin  = terminal_margin * ((year + 1) / T)
                nopat   = rev * margin * (1 - tax_rate)
                reinvestment = (growth * rev) / roic
                fcf     = nopat - reinvestment
                fcf_stream.append(fcf / (1 + wacc_test) ** (year + 1))

            terminal_nopat        = rev * terminal_margin * (1 - tax_rate)
            terminal_reinvestment = (terminal_growth * rev) / roic
            terminal_fcf          = terminal_nopat - terminal_reinvestment
            tv                    = terminal_fcf / (wacc_test - terminal_growth)
            tv_discounted         = tv / (1 + wacc_test) ** T

            ev_temp.append(sum(fcf_stream) + tv_discounted)

        sensitivity_matrix[i, j] = np.mean(np.array(ev_temp) > deal_price)

# Sensitivity Heatmap
plt.figure(figsize=(10, 7))
plt.imshow(
    sensitivity_matrix,
    extent=[wacc_range[0], wacc_range[-1], margin_range[0], margin_range[-1]],
    aspect='auto',
    origin='lower',
    cmap='RdYlGn'
)
plt.colorbar(label="Probability Deal Pays Off")
plt.xlabel("WACC")
plt.ylabel("Terminal EBIT Margin")
plt.title("Payoneer Payoff Probability — WACC vs Terminal Margin Sensitivity")
plt.tight_layout()
plt.show()