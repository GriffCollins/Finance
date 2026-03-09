import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

tickers = ["NEM", "PRIM", "LDO.MI", "TTWO", "ILMN", "DVN"]
start_date = "2024-01-01"
end_date = datetime.today().strftime("%Y-%m-%d")

data = yf.download(
    tickers,
    start=start_date,
    end=end_date,
    progress=False
)

prices = data["Close"]
plt.figure(figsize=(12, 6))

for ticker in tickers:
    plt.plot(prices.index, prices[ticker], label=ticker)

plt.title("Stock Adjusted Closing Prices Since Jan 1")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()