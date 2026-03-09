"""
PCA Decomposition Dashboard — 50-Stock Universe
install: pip install yfinance numpy pandas scikit-learn matplotlib seaborn
"""

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

# Universe definition with sector labels
universe = {
    "Technology"    : ["AAPL", "MSFT", "AVGO", "TXN", "AMAT", "FTNT"],
    "Comm Services" : ["GOOGL", "META", "VZ", "DIS"],
    "Cons Discret"  : ["AMZN", "HD", "MCD", "DHI", "TSLA"],
    "Cons Staples"  : ["PG", "KO", "COST", "CL"],
    "Energy"        : ["XOM", "CVX", "SLB", "EOG", "MPC"],
    "Financials"    : ["JPM", "BAC", "BRK-B", "AXP", "PGR"],
    "Health Care"   : ["JNJ", "LLY", "UNH", "ABT", "ISRG", "VRTX"],
    "Industrials"   : ["HON", "UPS", "CAT", "GE", "FAST"],
    "Materials"     : ["LIN", "APD", "NEM", "PKG"],
    "Real Estate"   : ["PLD", "AMT", "EQR"],
    "Utilities"     : ["NEE", "DUK", "SO"],
}
stocklist = [t for tickers in universe.values() for t in tickers]
sector_map = {t: s for s, tickers in universe.items() for t in tickers}

# Download 5 years of daily prices and compute returns
print("downloading price data...")
raw = yf.download(stocklist, period="5y", auto_adjust=True,
                  group_by="ticker", threads=True, progress=False)
prices  = pd.DataFrame({t: raw[t]["Close"] for t in stocklist if t in raw})
prices  = prices.loc[:, prices.isna().mean() < 0.05].ffill().dropna()
returns = prices.pct_change().dropna()
loaded  = list(returns.columns)
print(f"  {len(loaded)} assets  x  {len(returns)} days\n")

# Standardise returns before fitting pca (zero mean, unit variance)
scaler       = StandardScaler()
returns_std  = scaler.fit_transform(returns.values)

# Fit full pca on the standardised return matrix
pca      = PCA()
scores   = pca.fit_transform(returns_std)   # shape: (T, n_assets)
loadings = pca.components_.T               # shape: (n_assets, n_components)
evr      = pca.explained_variance_ratio_   # proportion of variance per component
cumvar   = np.cumsum(evr)                  # cumulative explained variance

# Identify how many components reach 80% and 90% explained variance thresholds
n80 = int(np.searchsorted(cumvar, 0.80)) + 1
n90 = int(np.searchsorted(cumvar, 0.90)) + 1
print(f"  components to explain 80% variance: {n80}")
print(f"  components to explain 90% variance: {n90}")
print(f"  pc1 explains: {evr[0]:.1%}  pc2: {evr[1]:.1%}  pc3: {evr[2]:.1%}\n")

# Build sector colour palette — one distinct colour per sector
sector_list   = list(universe.keys())
sector_colors = plt.cm.tab20(np.linspace(0, 1, len(sector_list)))
color_map     = dict(zip(sector_list, sector_colors))
stock_colors  = [color_map[sector_map[t]] for t in loaded]

# Convert PC scores to a time-indexed dataframe for rolling analysis
scores_df = pd.DataFrame(scores[:, :5],
                          index=returns.index,
                          columns=[f"pc{i+1}" for i in range(5)])

# Rolling 63-day variance explained by pc1 (proxy for systemic risk concentration)
roll_pc1_var = scores_df["pc1"].rolling(63).var()
roll_total_var = pd.DataFrame(returns_std).rolling(63).var().mean(axis=1)
roll_pc1_share = (roll_pc1_var / roll_total_var.values).rolling(5).mean()

# Build a 4x3 dashboard grid — tight, dark, editorial aesthetic
bg    = "#0a0a0f"
panel = "#12121a"
grid_c = "#1e1e2e"
acc1  = "#e8c547"   # amber — primary accent
acc2  = "#5bc4e8"   # cyan — secondary
acc3  = "#e87c5b"   # coral — tertiary
acc4  = "#7be89a"   # mint — quaternary
txt   = "#d4d4e8"
dim   = "#666680"

plt.rcParams.update({
    "font.family"       : "monospace",
    "axes.facecolor"    : panel,
    "figure.facecolor"  : bg,
    "text.color"        : txt,
    "axes.labelcolor"   : dim,
    "xtick.color"       : dim,
    "ytick.color"       : dim,
    "axes.edgecolor"    : grid_c,
    "grid.color"        : grid_c,
    "grid.linewidth"    : 0.4,
})

fig = plt.figure(figsize=(22, 18), facecolor=bg)
fig.suptitle("PCA DECOMPOSITION  ·  50-STOCK UNIVERSE  ·  5Y DAILY RETURNS",
             fontsize=11, color=acc1, fontweight="bold",
             y=0.98, fontfamily="monospace")

gs = gridspec.GridSpec(4, 3, figure=fig,
                       hspace=0.52, wspace=0.32,
                       top=0.95, bottom=0.04, left=0.06, right=0.97)

def style_ax(ax, title, grid=True):
    ax.set_title(title, fontsize=8, color=acc1, pad=6,
                 fontfamily="monospace", loc="left")
    ax.tick_params(labelsize=7)
    if grid:
        ax.grid(True, alpha=0.4)
    for spine in ax.spines.values():
        spine.set_edgecolor(grid_c)


# Panel 1: scree plot — eigenvalue bar chart with cumulative line
ax1 = fig.add_subplot(gs[0, 0])
n_show = 15
bars = ax1.bar(range(1, n_show + 1), evr[:n_show] * 100,
               color=acc1, alpha=0.75, width=0.7)
ax1_r = ax1.twinx()
ax1_r.plot(range(1, n_show + 1), cumvar[:n_show] * 100,
           color=acc2, lw=1.8, marker="o", ms=3)
ax1_r.axhline(80, color=acc3, lw=0.8, ls="--", alpha=0.7, label="80%")
ax1_r.axhline(90, color=acc4, lw=0.8, ls="--", alpha=0.7, label="90%")
ax1_r.tick_params(labelsize=7, colors=dim)
ax1_r.set_ylabel("cumulative %", fontsize=7, color=dim)
ax1.set_xlabel("component", fontsize=7)
ax1.set_ylabel("variance explained %", fontsize=7)
style_ax(ax1, "scree plot")

# Panel 2: biplot — stocks projected onto pc1 vs pc2, coloured by sector
ax2 = fig.add_subplot(gs[0, 1])
scale = 3.5
for i, ticker in enumerate(loaded):
    ax2.scatter(loadings[i, 0] * scale, loadings[i, 1] * scale,
                color=color_map[sector_map[ticker]], s=28, zorder=3, alpha=0.9)
    ax2.annotate(ticker, (loadings[i, 0] * scale, loadings[i, 1] * scale),
                 fontsize=5.5, color=txt, alpha=0.85,
                 xytext=(3, 2), textcoords="offset points")
ax2.axhline(0, color=grid_c, lw=0.6)
ax2.axvline(0, color=grid_c, lw=0.6)
ax2.set_xlabel(f"pc1  ({evr[0]:.1%})", fontsize=7)
ax2.set_ylabel(f"pc2  ({evr[1]:.1%})", fontsize=7)
style_ax(ax2, "biplot  pc1 vs pc2")

# Sector legend embedded in biplot panel
for s, c in color_map.items():
    ax2.scatter([], [], color=c, s=18, label=s)
ax2.legend(fontsize=5, loc="lower right", framealpha=0.15,
           markerscale=1.2, labelcolor=txt)

# Panel 3: pc1 loadings bar — sorted by loading magnitude, coloured by sector
ax3 = fig.add_subplot(gs[0, 2])
pc1_load = pd.Series(loadings[:, 0], index=loaded).sort_values()
bar_cols  = [color_map[sector_map[t]] for t in pc1_load.index]
ax3.barh(pc1_load.index, pc1_load.values, color=bar_cols, alpha=0.8, height=0.7)
ax3.axvline(0, color=txt, lw=0.6)
ax3.tick_params(axis="y", labelsize=5.5)
ax3.set_xlabel("loading", fontsize=7)
style_ax(ax3, "pc1 loadings  (market factor)", grid=False)

# Panel 4: pc2 loadings bar — reveals growth vs defensive split
ax4 = fig.add_subplot(gs[1, 0])
pc2_load = pd.Series(loadings[:, 1], index=loaded).sort_values()
bar_cols2 = [color_map[sector_map[t]] for t in pc2_load.index]
ax4.barh(pc2_load.index, pc2_load.values, color=bar_cols2, alpha=0.8, height=0.7)
ax4.axvline(0, color=txt, lw=0.6)
ax4.tick_params(axis="y", labelsize=5.5)
ax4.set_xlabel("loading", fontsize=7)
style_ax(ax4, "pc2 loadings  (growth vs defensive)", grid=False)

# Panel 5: pc3 loadings bar
ax5 = fig.add_subplot(gs[1, 1])
pc3_load = pd.Series(loadings[:, 2], index=loaded).sort_values()
bar_cols3 = [color_map[sector_map[t]] for t in pc3_load.index]
ax5.barh(pc3_load.index, pc3_load.values, color=bar_cols3, alpha=0.8, height=0.7)
ax5.axvline(0, color=txt, lw=0.6)
ax5.tick_params(axis="y", labelsize=5.5)
ax5.set_xlabel("loading", fontsize=7)
style_ax(ax5, "pc3 loadings  (sector rotation)", grid=False)

# Panel 6: cumulative variance by sector — how much each sector contributes
ax6 = fig.add_subplot(gs[1, 2])
sector_var = {}
for s, tickers in universe.items():
    tks_loaded = [t for t in tickers if t in loaded]
    if not tks_loaded:
        continue
    idx    = [loaded.index(t) for t in tks_loaded]
    # variance of the sector's average return series
    sector_ret = returns[tks_loaded].mean(axis=1)
    sector_var[s] = sector_ret.var()
sv = pd.Series(sector_var).sort_values(ascending=True)
ax6.barh(sv.index, sv.values * 1e4,
         color=[color_map[s] for s in sv.index], alpha=0.8, height=0.7)
ax6.set_xlabel("variance  (x 10⁻⁴)", fontsize=7)
ax6.tick_params(axis="y", labelsize=6.5)
style_ax(ax6, "sector return variance", grid=False)

# Panel 7: pc1 score time series — systemic risk factor over time
ax7 = fig.add_subplot(gs[2, :2])
ax7.plot(scores_df.index, scores_df["pc1"],
         color=acc1, lw=0.7, alpha=0.85, label="pc1 score")
ax7.fill_between(scores_df.index, scores_df["pc1"], 0,
                 where=scores_df["pc1"] < 0,
                 color=acc3, alpha=0.25, label="negative (market stress)")
ax7.fill_between(scores_df.index, scores_df["pc1"], 0,
                 where=scores_df["pc1"] > 0,
                 color=acc4, alpha=0.15, label="positive")
ax7.axhline(0, color=grid_c, lw=0.6)
ax7.set_ylabel("score", fontsize=7)
ax7.legend(fontsize=7, framealpha=0.15, loc="upper left")
style_ax(ax7, "pc1 score over time  (market factor)")

# Panel 8: rolling pc1 variance share — systemic risk concentration
ax8 = fig.add_subplot(gs[2, 2])
ax8.plot(roll_pc1_share.index, roll_pc1_share.values,
         color=acc2, lw=1.0, alpha=0.9)
ax8.fill_between(roll_pc1_share.index, roll_pc1_share.values,
                 roll_pc1_share.median(),
                 where=roll_pc1_share.values > roll_pc1_share.median(),
                 alpha=0.2, color=acc3, label="above median (high systemic risk)")
ax8.axhline(roll_pc1_share.median(), color=txt, lw=0.6, ls=":", alpha=0.5)
ax8.set_ylabel("pc1 variance share", fontsize=7)
ax8.legend(fontsize=6.5, framealpha=0.15)
style_ax(ax8, "rolling 63d systemic risk concentration")

# Panel 9: pc1 vs pc2 score scatter coloured by time (early=dark, late=bright)
ax9 = fig.add_subplot(gs[3, 0])
n_pts = len(scores_df)
time_colors = plt.cm.plasma(np.linspace(0.1, 0.95, n_pts))
ax9.scatter(scores_df["pc1"], scores_df["pc2"],
            c=np.linspace(0, 1, n_pts), cmap="plasma", s=1.5, alpha=0.6)
ax9.axhline(0, color=grid_c, lw=0.4)
ax9.axvline(0, color=grid_c, lw=0.4)
ax9.set_xlabel(f"pc1 score", fontsize=7)
ax9.set_ylabel(f"pc2 score", fontsize=7)
style_ax(ax9, "regime scatter  pc1 vs pc2  (dark=early  bright=recent)")

# Panel 10: heatmap of top-10 stock loadings across first 5 pcs
ax10 = fig.add_subplot(gs[3, 1])
top_stocks_idx = np.argsort(np.abs(loadings[:, 0]))[::-1][:12]
top_tickers    = [loaded[i] for i in top_stocks_idx]
top_loadings   = loadings[top_stocks_idx, :5]
cmap_div = LinearSegmentedColormap.from_list("rg",
           [acc3, bg, acc4], N=256)
im = ax10.imshow(top_loadings, aspect="auto", cmap=cmap_div,
                 vmin=-0.25, vmax=0.25)
ax10.set_xticks(range(5))
ax10.set_xticklabels([f"pc{i+1}" for i in range(5)], fontsize=7)
ax10.set_yticks(range(len(top_tickers)))
ax10.set_yticklabels(top_tickers, fontsize=6.5)
plt.colorbar(im, ax=ax10, fraction=0.03, pad=0.02).ax.tick_params(labelsize=6)
style_ax(ax10, "loading heatmap  top-12 stocks  x  first 5 pcs", grid=False)

# Panel 11: pc2 and pc3 scores over time — sector rotation dynamics
ax11 = fig.add_subplot(gs[3, 2])
ax11.plot(scores_df.index, scores_df["pc2"],
          color=acc2, lw=0.7, alpha=0.85, label="pc2 (growth/defensive)")
ax11.plot(scores_df.index, scores_df["pc3"],
          color=acc3, lw=0.7, alpha=0.7, label="pc3 (sector rotation)")
ax11.axhline(0, color=grid_c, lw=0.5)
ax11.legend(fontsize=6.5, framealpha=0.15)
style_ax(ax11, "pc2 and pc3 scores over time")

plt.savefig("pca_dashboard.png", dpi=180, bbox_inches="tight", facecolor=bg)
print("dashboard saved to pca_dashboard.png")
plt.show()