import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf

stocklist = ["NEM", 'PRIM','LDO.MI', 'TTWO', 'ILMN', 'ADYEN.AS', 'DVN']
df = yf.download(tickers=stocklist, auto_adjust=True)
returns = df["Close"].pct_change().dropna()

# Correlation matrix
corr = returns.corr()

# Mask upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool))

plt.figure(figsize=(8, 6))
sns.heatmap(
    corr,
    mask=mask,
    cmap="coolwarm",
    vmin=-1,
    vmax=1,
    center=0,
    annot=True,
    fmt=".2f",
    linewidths=0.5,
    cbar_kws={"shrink": 0.8}
)

plt.title("Asset Correlation Matrix (Lower Triangle)")
plt.tight_layout()
plt.show()

eigvals = np.linalg.eigvalsh(corr.values)
eigvals = np.sort(eigvals)[::-1]

plt.plot(eigvals, marker='o')
plt.title("Eigenvalues of Correlation Matrix")
plt.ylabel("Eigenvalue")
plt.xlabel("Component")
plt.grid()
plt.show()