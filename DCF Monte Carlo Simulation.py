import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import fredapi
import pandas as pd

payo          = yf.Ticker("PAYO")
financials    = payo.financials
cashflow      = payo.cashflow
balance_sheet = payo.balance_sheet
info          = payo.info

print("=" * 60)
print("PAYONEER DIAGNOSTICS")
print("=" * 60)

print("\nFREE CASH FLOW (most recent first):")
fcf = cashflow.loc['Free Cash Flow']
print(fcf)

print("\nTOTAL REVENUE:")
revenue = financials.loc['Total Revenue']
print(revenue)

print("\nYOY REVENUE GROWTH RATES:")
growth_rates = revenue.pct_change(periods=-1)
print(growth_rates)

print(f"\nBETA:       {info.get('beta', 'N/A')}")
print(f"MARKET CAP: {info.get('marketCap', 'N/A'):,}")

try:
    total_debt = balance_sheet.loc['Total Debt'].iloc[0]
    print(f"TOTAL DEBT: {total_debt:,}")
except KeyError:
    total_debt = 0
    print("TOTAL DEBT: Not found, defaulting to 0")

tax_provision = financials.loc['Tax Provision'].iloc[0]
pretax_income = financials.loc['Pretax Income'].iloc[0]
if pretax_income > 0:
    tax_rate = tax_provision / pretax_income
    print(f"\nEffective Tax Rate (yfinance):  {tax_rate:.4f}")
else:
    tax_rate = 0.156  # from accretion/dilution model
    print(f"\nPretax income negative — using A/D model tax rate: {tax_rate:.4f}")

FRED_API_KEY = "f022be36d1b0a0ad0437176d04939ad8"
fred         = fredapi.Fred(api_key=FRED_API_KEY)

print("\n" + "=" * 60)
print("FRED DATA")
print("=" * 60)

# Risk free rate
rf_series = fred.get_series('DGS10')
risk_free  = rf_series.dropna().iloc[-1] / 100
print(f"Risk Free Rate (10yr Treasury): {risk_free:.4f}")

# Market return and premium
sp500          = fred.get_series('SP500')
sp500_clean    = sp500.dropna()
market_return  = sp500_clean.pct_change(periods=12).dropna().mean() * 12
market_premium = market_return - risk_free
print(f"Market Return (annualised):     {market_return:.4f}")
print(f"Market Risk Premium:            {market_premium:.4f}")

# Cost of debt — use 7% from A/D model, BAA as sanity check
baa_series        = fred.get_series('BAA')
cost_of_debt_baa  = baa_series.dropna().iloc[-1] / 100
cost_of_debt      = 0.07  # from accretion/dilution model debt assumptions
print(f"Cost of Debt (A/D model):       {cost_of_debt:.4f}")
print(f"Cost of Debt (BAA sanity check):{cost_of_debt_baa:.4f}")

# Terminal growth
breakeven_inflation = fred.get_series('T10YIE').dropna().iloc[-1] / 100
gdp_series          = fred.get_series('A191RL1Q225SBEA')
long_run_gdp        = gdp_series.dropna().tail(20).mean() / 100
terminal_growth     = 0.5 * breakeven_inflation + 0.5 * long_run_gdp
print(f"\nBreakeven Inflation:            {breakeven_inflation:.4f}")
print(f"Long Run GDP Growth:            {long_run_gdp:.4f}")
print(f"Terminal Growth Rate:           {terminal_growth:.4f}")

print("\n" + "=" * 60)
print("WACC CONSTRUCTION")
print("=" * 60)

beta           = min(info['beta'], 2.0)  # cap at 2.0 — yfinance unreliable for small caps
cost_of_equity = risk_free + beta * market_premium
total_equity   = info['marketCap']
weight_equity  = total_equity / (total_equity + total_debt)
weight_debt    = total_debt   / (total_equity + total_debt)
WACC_mean      = (weight_equity * cost_of_equity) + (weight_debt * cost_of_debt * (1 - tax_rate))

baa_clean  = baa_series.dropna()
baa_recent = baa_clean.loc[baa_clean.index >= (baa_clean.index[-1] - pd.DateOffset(years=1))]
WACC_std   = (baa_recent.std() / 100) * weight_debt + 0.005

print(f"Beta (capped):                  {beta:.4f}")
print(f"Cost of Equity (CAPM):          {cost_of_equity:.4f}")
print(f"Cost of Debt:                   {cost_of_debt:.4f}")
print(f"Weight Equity:                  {weight_equity:.4f}")
print(f"Weight Debt:                    {weight_debt:.4f}")
print(f"Calculated WACC:                {WACC_mean:.4f}")
print(f"WACC Std (data-driven):         {WACC_std:.4f}")

print("\n" + "=" * 60)
print("FCF PARAMETERS")
print("=" * 60)

initial_FCFE    = cashflow.loc['Free Cash Flow'].iloc[0]
growth_FCFE     = growth_rates.mean()
std_growth_FCFE = growth_rates.std()

print(f"Initial FCFE:                   {initial_FCFE:,.2f}")
print(f"Mean Growth:                    {growth_FCFE:.4f}")
print(f"Growth Std:                     {std_growth_FCFE:.4f}")

print("\n" + "=" * 60)
print("MONTE CARLO SIMULATION")
print("=" * 60)

num_scenarios   = 100000
forecast_period = 5
alpha           = 0.05  # significance level for CI — not CAPM alpha

wacc_scenarios = np.random.normal(WACC_mean, WACC_std, num_scenarios)

future_fcfe         = np.zeros((num_scenarios, forecast_period))
future_fcfe[:, 0]   = initial_FCFE * (1 + np.random.normal(growth_FCFE, std_growth_FCFE, num_scenarios))
for t in range(1, forecast_period):
    future_fcfe[:, t] = future_fcfe[:, t-1] * (1 + np.random.normal(growth_FCFE, std_growth_FCFE, num_scenarios))

present_values = []
for scenario, wacc in zip(future_fcfe, wacc_scenarios):
    pv = sum(fcfe / (1 + wacc) ** (t + 1) for t, fcfe in enumerate(scenario))
    terminal_value = (scenario[-1] * (1 + terminal_growth)) / (wacc - terminal_growth)
    pv += terminal_value / (1 + wacc) ** forecast_period
    present_values.append(pv)

present_values = np.array(present_values)

# Results
company_value        = np.mean(present_values)
lower_bound, upper_bound = np.percentile(present_values, [100 * alpha / 2, 100 * (1 - alpha / 2)])

print(f"Estimated Company Value:        {company_value:,.2f}")
print(f"95% Confidence Interval:        [{lower_bound:,.2f}, {upper_bound:,.2f}]")

print("\n" + "=" * 60)
print("DEAL PAYOFF ANALYSIS")
print("=" * 60)

deal_price      = 3_386_180_000  # $4,372.80M from sources & uses
scenarios_above = np.sum(present_values > deal_price)
pct_payoff      = (scenarios_above / num_scenarios) * 100

print(f"Deal Price:                     ${deal_price:,.0f}")
print(f"Scenarios that pay off:         {scenarios_above:,} / {num_scenarios:,}")
print(f"Probability deal pays off:      {pct_payoff:.1f}%")
print(f"Deal price percentile:          {100 - pct_payoff:.1f}th percentile")

# Plot 1: Monte Carlo Distribution
plt.figure(figsize=(12, 8))
plt.hist(present_values, bins=50, alpha=0.5, color='blue')
plt.title(f"Payoneer Monte Carlo DCF — {forecast_period} Year Forecast ({num_scenarios:,} Scenarios)")
plt.axvline(company_value, color='red',    linestyle='--', label=f"Mean:                    {company_value:,.0f}")
plt.axvline(upper_bound,   color='orange', linestyle='--', label=f"95% CI Upper:          {upper_bound:,.0f}")
plt.axvline(lower_bound,   color='green',  linestyle='--', label=f"95% CI Lower:           {lower_bound:,.0f}")
plt.axvline(deal_price,    color='white',  linestyle='-',  linewidth=2,
            label=f"Deal Price: ${deal_price/1e9:.2f}B  |  Payoff probability: {pct_payoff:.1f}%")
plt.xlabel("Valuation ($)")
plt.ylabel("Frequency")
plt.legend()
plt.tight_layout()
plt.show()

# Plot 2: Parameter Space Contour
T      = forecast_period
sigma  = 3
w_sd   = WACC_std
f_sd   = std_growth_FCFE * abs(initial_FCFE)

w_vals = np.linspace(WACC_mean - sigma * w_sd, WACC_mean + sigma * w_sd, 33 * sigma)
f_vals = np.linspace(initial_FCFE - sigma * f_sd, initial_FCFE + sigma * f_sd, 33 * sigma)

W, F = np.meshgrid(w_vals, f_vals)
V    = np.zeros_like(W)

for i in range(W.shape[0]):
    for j in range(W.shape[1]):
        pv = sum(F[i, j] / (1 + W[i, j]) ** k for k in range(1, T + 1))
        tv = (F[i, j] * (1 + terminal_growth)) / (W[i, j] - terminal_growth)
        V[i, j] = pv + tv / (1 + W[i, j]) ** T

plt.figure(figsize=(9, 7))
contours = plt.contourf(W, F, V, levels=50, cmap='plasma')
plt.colorbar(contours, label='DCF Value ($)')
plt.scatter(WACC_mean, initial_FCFE, color='white', zorder=5, s=100,
            label=f'MC Mean\nWACC={WACC_mean:.3f}, FCF={initial_FCFE:,.0f}')
plt.xlabel('Discount Rate (WACC)')
plt.ylabel('Initial Free Cash Flow ($)')
plt.title(f'Payoneer DCF Parameter Space — {sigma}σ Sensitivity')
plt.legend(facecolor='black', labelcolor='white')
plt.tight_layout()
plt.show()