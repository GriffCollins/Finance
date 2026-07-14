import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ticker = yf.Ticker("AAPL")

cash_flow = ticker.cashflow
income_statement = ticker.income_stmt
balance_sheet = ticker.balance_sheet

cfo = cash_flow.loc['Cash Flow From Continuing Operating Activities']
net_income = income_statement.loc['Net Income']
total_assets = balance_sheet.loc['Total Assets']

sloan_accrual = (net_income - cfo) / total_assets

plt.plot(sloan_accrual)
plt.show()
