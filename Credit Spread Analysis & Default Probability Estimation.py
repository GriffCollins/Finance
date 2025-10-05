import pandas as pd
from fredapi import Fred
import matplotlib.pyplot as plt
import yfinance as yf
import numpy as np
from scipy.stats import norm
from alpha_vantage.fundamentaldata import FundamentalData

ALPHA_VANTAGE_API_KEY = "HDYUIAIHEUABSU8H"
FRED_API_KEY = "f022be36d1b0a0ad0437176d04939ad8"
fred = Fred(api_key=FRED_API_KEY)
series = {
    "bbb_yield": "BAMLC0A4CBBBEY",
    "aaa_yield": "BAMLC0A1CAAAEY",
    "treasury_10y": "DGS10"
}

data_dict = {}
for name, code in series.items():
    data_dict[name] = fred.get_series(code)

data = pd.DataFrame(data_dict)
data["bbb_yield"] /= 100
data["aaa_yield"] /= 100
data["treasury_10y"] /= 100
historical_default_5y = 0.0211

#Results in large swings due to compounding nature
def default_est(data, bond_name):
    data[bond_name + "_spread"] = data[bond_name] - data["treasury_10y"]
    data["rolling_" + bond_name + "_5y"] = data[bond_name + "_spread"].rolling(252*5, min_periods=1).mean()
    data[bond_name + "_est"] = historical_default_5y * data[bond_name] / data["rolling_" + bond_name + "_5y"]
    return data[bond_name + "_est"]

def hazard_rate(data, bond_name, recovery_rate):
    data[bond_name + "_spread"] = data[bond_name] - data["treasury_10y"]
    data["rolling_" + bond_name + "_5y"] = data[bond_name + "_spread"].rolling(252*5, min_periods=1).mean()
    data[bond_name + "_lambda"] = data["rolling_" + bond_name + "_5y"] / (1 - recovery_rate)
    return data[bond_name + "_lambda"]


def get_debt_data(ticker):
    fd = FundamentalData(ALPHA_VANTAGE_API_KEY)
    balance_sheet = fd.get_balance_sheet_annual(symbol=ticker)[0]

    print("Available columns in balance sheet:", balance_sheet.columns.tolist())

    debt_columns = [
        'totalDebt',
        'totalLiabilities',
        'shortLongTermDebtTotal',
        'longTermDebt',
        'shortTermDebt']

    for col in debt_columns:
        if col in balance_sheet.columns:
            total_debt = balance_sheet[col].iloc[0]
            print(f"Using debt column: {col}")
            return parse_financial_value(total_debt)

    raise KeyError(f"No recognized debt column found. Available columns: {balance_sheet.columns}")

def get_equity_data(ticker):
    stock = yf.Ticker(ticker)
    E = stock.info['marketCap']
    hist_prices = stock.history(period="1y")['Close']
    log_returns = np.log(hist_prices / hist_prices.shift(1))
    sigma_E = log_returns.std() * np.sqrt(252)
    return E, sigma_E

def get_risk_free_rate():
    return fred.get_series(series["treasury_10y"]).iloc[-1] / 100

def parse_financial_value(value_str):
    if isinstance(value_str, str):
        if 'B' in value_str:
            return float(value_str.replace('B', '')) * 1e9  # Billion
        elif 'M' in value_str:
            return float(value_str.replace('M', '')) * 1e6  # Million
        else:
            return float(value_str)
    else:
        return value_str

def merton_model(E, D, T, r, sigma_E, max_iter=1000, tol=1e-6):
    E = float(E)
    D = float(D)
    V_A = E + D
    sigma_A = sigma_E * (E / V_A)

    for _ in range(max_iter):
        d1 = (np.log(V_A / D) + (r + 0.5 * sigma_A ** 2) * T) / (sigma_A * np.sqrt(T))
        d2 = d1 - sigma_A * np.sqrt(T)
        E_new = V_A * norm.cdf(d1) - D * np.exp(-r * T) * norm.cdf(d2)

        if abs(E_new - E) < tol:
            break

        derivative = norm.cdf(d1)
        V_A = V_A - (E_new - E) / derivative
        sigma_A = sigma_E * (E / V_A)

    PD = norm.cdf(-d2)
    return PD

plt.title("Historical Scaling for AAA and BBB Default Estimate")
plt.plot(data.index, default_est(data, "bbb_yield"), label="BBB")
plt.plot(data.index, default_est(data, "aaa_yield"), label="AAA")
plt.legend()
plt.ylim(-0.02, 1)
plt.show()

plt.title("Hazard Rate for AAA and BBB bonds")
plt.plot(data.index, hazard_rate(data, "bbb_yield", 0.4), label="BBB")
plt.plot(data.index, hazard_rate(data, "aaa_yield", 0.4), label="AAA")
plt.legend()
plt.show()

ticker = "F"
E, sigma_E = get_equity_data(ticker)
D = get_debt_data(ticker)
r = get_risk_free_rate()

# Compute PD
T = 1  # 1-year horizon
PD = merton_model(E, D, T, r, sigma_E)

print("\n--- Merton Model Results ---")
print(f"Equity Value (E): ${E/1e9:.2f}B")
print(f"Total Debt (D): ${D/1e9:.2f}B")
print(f"Equity Volatility (σ_E): {sigma_E:.2%}")
print(f"Risk-Free Rate (r): {r:.2%}")
print(f"Probability of Default (PD) in 1 year: {PD:.2%}")





