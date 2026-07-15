import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter
import scipy.stats as stats
import yfinance as yf
from datetime import timedelta

# ==========================================
# CONFIGURATION
# ==========================================

tickers = ["NEM", "PRIM", "LDO.MI", "TTWO", "ILMN", "ADYEN.AS", "DVN"]

weights = np.array([
    0.1630,  # NEM
    0.1819,  # PRIM
    0.0887,  # LDO.MI
    0.1880,  # TTWO
    0.1396,  # ILMN
    0.0388,  # ADYEN.AS
    0.2000  # DVN
])

risk_free_rate = 0.05
daily_rf = risk_free_rate / 252

estimation_start = "2023-12-01"
estimation_end = "2025-12-01"
simulation_start = "2025-12-02"
simulation_end = "2026-05-01"

# ==========================================
# VALIDATION
# ==========================================

assert len(weights) == len(tickers)
assert abs(weights.sum() - 1) < 1e-6

# ==========================================
# DOWNLOAD
# ==========================================

print("\nDOWNLOADING DATA")

benchmark_ticker = "URTH"

# Download portfolio tickers and the benchmark in a single call
prices_all = yf.download(
    tickers + [benchmark_ticker],
    start=estimation_start,
    end=simulation_end,
    auto_adjust=True
)["Close"]

# Calculate portfolio returns, dropping rows where any portfolio asset is missing
prices_portfolio = prices_all[tickers]
returns_portfolio_all = prices_portfolio.pct_change().dropna()

# Extract benchmark returns separately to avoid losing portfolio trading days
returns_urth_all = prices_all[benchmark_ticker].pct_change().dropna()

returns_opt = returns_portfolio_all.loc[estimation_start:estimation_end].copy()
returns_sim = returns_portfolio_all.loc[simulation_start:simulation_end].copy()
returns_urth_sim = returns_urth_all.loc[simulation_start:simulation_end].copy()

print("Optimization:", returns_opt.index[0].date(), "to", returns_opt.index[-1].date())
print("Simulation:  ", returns_sim.index[0].date(), "to", returns_sim.index[-1].date())


# ==========================================
# SIMULATOR
# ==========================================

class PortfolioSimulator:

    def __init__(self, weights, returns):
        self.position_capital = weights.copy()
        self.returns = returns

    def run(self):
        portfolio = [1.0]
        for t in range(len(self.returns)):
            r = self.returns.iloc[t].values
            self.position_capital *= (1 + r)
            portfolio.append(self.position_capital.sum())
        return np.array(portfolio)


print("\nRUNNING SIMULATION")
sim = PortfolioSimulator(weights, returns_sim)
portfolio = sim.run()


# ==========================================
# COMPUTE ALL STATISTICS
# ==========================================

def compute_stats(series, returns_df, w, tickers, benchmark_returns=None, risk_free_rate=0.05):
    r = pd.Series(series).pct_change().dropna()
    r.index = returns_df.index
    n = len(r)

    # --- Returns ---
    total_return = series[-1] - 1
    annualised_ret = (1 + total_return) ** (252 / n) - 1
    annual_vol = r.std() * np.sqrt(252)
    monthly_ret = (1 + total_return) ** (21 / n) - 1

    # --- Risk-adjusted ---
    excess = r - daily_rf
    sharpe = excess.mean() / r.std() * np.sqrt(252) if r.std() > 0 else np.nan

    downside_r = r[r < 0]
    downside_std = downside_r.std() * np.sqrt(252) if len(downside_r) > 1 else np.nan
    sortino = (r.mean() * 252 - risk_free_rate) / downside_std if downside_std else np.nan

    omega_thresh = 0.0
    gains = r[r > omega_thresh].sum()
    losses = abs(r[r < omega_thresh].sum())
    omega = gains / losses if losses > 0 else np.nan

    # --- Benchmark & Regression (Beta/Alpha) ---
    if benchmark_returns is not None:
        common_idx = r.index.intersection(benchmark_returns.index)
        r_aligned = r.loc[common_idx]
        bench_aligned = benchmark_returns.loc[common_idx]

        # Simple linear regression on realized daily return series
        slope, intercept, r_value, p_value, std_err = stats.linregress(bench_aligned, r_aligned)
        beta = slope
        alpha_daily = intercept
        alpha_ann = (1 + alpha_daily) ** 252 - 1 if alpha_daily > -1 else alpha_daily * 252
        r_squared = r_value ** 2
    else:
        beta, alpha_ann, r_squared = np.nan, np.nan, np.nan

    # --- Drawdown ---
    peak = np.maximum.accumulate(series)
    dd_series = series / peak - 1
    max_dd = dd_series.min()
    calmar = annualised_ret / abs(max_dd) if max_dd != 0 else np.nan

    trough_idx = int(np.argmin(dd_series))
    post_trough = series[trough_idx:]
    recovery_level = peak[trough_idx]
    recovered = np.where(post_trough >= recovery_level)[0]
    recovery_days = int(recovered[0]) if len(recovered) > 0 else None

    # Average drawdown duration
    in_dd = dd_series < 0
    dd_lengths = []
    count = 0
    for val in in_dd:
        if val:
            count += 1
        else:
            if count > 0:
                dd_lengths.append(count)
            count = 0
    avg_dd_duration = np.mean(dd_lengths) if dd_lengths else 0

    # Ulcer index
    ulcer_index = np.sqrt(np.mean(dd_series ** 2))

    # --- Tail risk ---
    var_95 = np.percentile(r, 5)
    cvar_95 = r[r <= var_95].mean()
    var_99 = np.percentile(r, 1)
    cvar_99 = r[r <= var_99].mean()
    tail_ratio = abs(cvar_95 / var_95) if var_95 != 0 else np.nan

    # --- Distribution shape ---
    skewness = stats.skew(r)
    kurt = stats.kurtosis(r)  # excess kurtosis
    jb_stat, jb_p = stats.jarque_bera(r)

    # --- Win/loss profile ---
    win_rate = (r > 0).mean()
    avg_win = r[r > 0].mean() if (r > 0).any() else np.nan
    avg_loss = r[r < 0].mean() if (r < 0).any() else np.nan
    win_loss_ratio = abs(avg_win / avg_loss) if avg_loss and avg_loss != 0 else np.nan
    best_day = r.max()
    worst_day = r.min()
    best_week = r.rolling(5).sum().max()
    worst_week = r.rolling(5).sum().min()

    # --- Streak stats ---
    signs = np.sign(r.values)
    max_win_streak = max_lose_streak = cur = 0
    for s in signs:
        cur = cur + 1 if s > 0 else (0 if s <= 0 else cur)
        max_win_streak = max(max_win_streak, cur)
    cur = 0
    for s in signs:
        cur = cur + 1 if s < 0 else (0 if s >= 0 else cur)
        max_lose_streak = max(max_lose_streak, cur)

    # --- Per-ticker ---
    ticker_contribs = {}
    for i, t in enumerate(tickers):
        pos_return = (returns_df[t] + 1).prod() - 1
        ticker_contribs[t] = w[i] * pos_return

    return {
        "total_return": total_return,
        "annualised_ret": annualised_ret,
        "monthly_ret": monthly_ret,
        "annual_vol": annual_vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "omega": omega,
        "beta": beta,
        "alpha_ann": alpha_ann,
        "r_squared": r_squared,
        "max_dd": max_dd,
        "recovery_days": recovery_days,
        "avg_dd_duration": avg_dd_duration,
        "ulcer_index": ulcer_index,
        "var_95": var_95,
        "cvar_95": cvar_95,
        "var_99": var_99,
        "cvar_99": cvar_99,
        "tail_ratio": tail_ratio,
        "skewness": skewness,
        "kurtosis": kurt,
        "jb_stat": jb_stat,
        "jb_p": jb_p,
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "win_loss_ratio": win_loss_ratio,
        "best_day": best_day,
        "worst_day": worst_day,
        "best_week": best_week,
        "worst_week": worst_week,
        "max_win_streak": max_win_streak,
        "max_lose_streak": max_lose_streak,
        "dd_series": dd_series,
        "returns": r,
        "ticker_contribs": ticker_contribs,
    }


st = compute_stats(portfolio, returns_sim, weights, tickers, benchmark_returns=returns_urth_sim,
                   risk_free_rate=risk_free_rate)

# ==========================================
# PRINT SUMMARY
# ==========================================

recovery_str = f"{st['recovery_days']}d" if st['recovery_days'] is not None else "not recovered"
jb_sig = "yes" if st['jb_p'] < 0.05 else "no"

print(f"""
{'=' * 50}
  SIMULATION RESULTS
{'=' * 50}

  RETURNS
  {'Total Return':<28}: {st['total_return']:>10.2%}
  {'Annualised Return':<28}: {st['annualised_ret']:>10.2%}
  {'Monthly Return (equiv.)':<28}: {st['monthly_ret']:>10.2%}
  {'Annual Volatility':<28}: {st['annual_vol']:>10.2%}

  RISK-ADJUSTED & SYSTEMATIC RISK
  {'Sharpe Ratio':<28}: {st['sharpe']:>10.3f}
  {'Sortino Ratio':<28}: {st['sortino']:>10.3f}
  {'Calmar Ratio':<28}: {st['calmar']:>10.3f}
  {'Omega Ratio':<28}: {st['omega']:>10.3f}
  {'Strategy Beta (vs URTH)':<28}: {st['beta']:>10.3f}
  {'Alpha (ann. vs URTH)':<28}: {st['alpha_ann']:>10.2%}
  {'R-squared (vs URTH)':<28}: {st['r_squared']:>10.3f}

  DRAWDOWN
  {'Max Drawdown':<28}: {st['max_dd']:>10.2%}
  {'Recovery from Max DD':<28}: {recovery_str:>10}
  {'Avg Drawdown Duration':<28}: {st['avg_dd_duration']:>9.1f}d
  {'Ulcer Index':<28}: {st['ulcer_index']:>10.4f}

  TAIL RISK
  {'VaR 95% (daily)':<28}: {st['var_95']:>10.2%}
  {'CVaR 95% (daily)':<28}: {st['cvar_95']:>10.2%}
  {'VaR 99% (daily)':<28}: {st['var_99']:>10.2%}
  {'CVaR 99% (daily)':<28}: {st['cvar_99']:>10.2%}
  {'CVaR/VaR (tail ratio)':<28}: {st['tail_ratio']:>10.3f}

  DISTRIBUTION
  {'Skewness':<28}: {st['skewness']:>10.3f}
  {'Excess Kurtosis':<28}: {st['kurtosis']:>10.3f}
  {'Jarque-Bera non-normal?':<28}: {jb_sig:>10}  (p={st['jb_p']:.4f})

  WIN / LOSS PROFILE
  {'Win Rate':<28}: {st['win_rate']:>10.2%}
  {'Avg Win (daily)':<28}: {st['avg_win']:>10.2%}
  {'Avg Loss (daily)':<28}: {st['avg_loss']:>10.2%}
  {'Win/Loss Ratio':<28}: {st['win_loss_ratio']:>10.3f}
  {'Best Day':<28}: {st['best_day']:>10.2%}
  {'Worst Day':<28}: {st['worst_day']:>10.2%}
  {'Best Week':<28}: {st['best_week']:>10.2%}
  {'Worst Week':<28}: {st['worst_week']:>10.2%}
  {'Max Win Streak':<28}: {st['max_win_streak']:>9}d
  {'Max Lose Streak':<28}: {st['max_lose_streak']:>9}d

  PER-TICKER CONTRIBUTION
  {'Ticker':<14} {'Contribution':>12}
  {'-' * 28}""")

for t, c in st['ticker_contribs'].items():
    print(f"  {t:<14} {c:>11.2%}")
print(f"  {'Total':<14} {sum(st['ticker_contribs'].values()):>11.2%}")
print(f"{'=' * 50}\n")

# ==========================================
# DASHBOARD
# ==========================================

plot_dates = [returns_sim.index[0] - timedelta(days=1)] + list(returns_sim.index)
r = st['returns']
dd_series = st['dd_series']
pct_fmt = FuncFormatter(lambda x, _: f"{x:.0%}")
pct2_fmt = FuncFormatter(lambda x, _: f"{x:.1%}")

DARK = "#1a1a1a"
MID = "#2b2b2b"
LIGHT = "#3d3d3d"
TEXT = "#e0e0e0"
DIM = "#888888"
GREEN = "#2ecc71"
RED = "#e74c3c"
BLUE = "#4fa3e0"
AMBER = "#f0a500"

fig = plt.figure(figsize=(20, 22), facecolor=DARK)
fig.suptitle("exSIF Paper Portfolio  |  Dec 2025 – May 2026",
             color=TEXT, fontsize=15, fontweight="bold", y=0.995)

gs = gridspec.GridSpec(
    4, 3,
    figure=fig,
    hspace=0.45,
    wspace=0.32,
    top=0.97,
    bottom=0.04,
    left=0.06,
    right=0.97
)


def style_ax(ax, title=""):
    ax.set_facecolor(MID)
    for spine in ax.spines.values():
        spine.set_color(LIGHT)
    ax.tick_params(colors=DIM, labelsize=8)
    ax.xaxis.label.set_color(DIM)
    ax.yaxis.label.set_color(DIM)
    if title:
        ax.set_title(title, color=TEXT, fontsize=9, fontweight="bold", pad=6)
    ax.grid(True, color=LIGHT, linewidth=0.4, alpha=0.6)


# ------------------------------------------
# Row 0 col 0-2: Stat cards
# ------------------------------------------

ax_cards = fig.add_subplot(gs[0, :])
ax_cards.set_facecolor(DARK)
for spine in ax_cards.spines.values():
    spine.set_visible(False)
ax_cards.set_xticks([])
ax_cards.set_yticks([])

# Stat cards now dynamically expanded to include Strategy Beta & Alpha
cards = [
    ("Total Return", f"{st['total_return']:+.2%}", st['total_return'] >= 0),
    ("Ann. Return", f"{st['annualised_ret']:+.2%}", st['annualised_ret'] >= 0),
    ("Ann. Volatility", f"{st['annual_vol']:.2%}", None),
    ("Sharpe", f"{st['sharpe']:.3f}", st['sharpe'] >= 1),
    ("Sortino", f"{st['sortino']:.3f}", st['sortino'] >= 1),
    ("Strategy Beta", f"{st['beta']:.3f}", None),
    ("Alpha (Ann.)", f"{st['alpha_ann']:+.2%}", st['alpha_ann'] >= 0),
    ("Max DD", f"{st['max_dd']:.2%}", False),
    ("Win Rate", f"{st['win_rate']:.1%}", st['win_rate'] >= 0.5),
    ("Omega", f"{st['omega']:.3f}", st['omega'] >= 1),
]

nw = len(cards)
pad = 0.01
cw = (1.0 - pad * (nw + 1)) / nw
ch = 0.80
cy = 0.10

for i, (label, value, good) in enumerate(cards):
    cx = pad + i * (cw + pad)
    color = GREEN if good is True else (RED if good is False else AMBER)
    if label == "Strategy Beta":
        color = BLUE
    rect = plt.Rectangle((cx, cy), cw, ch,
                         transform=ax_cards.transAxes,
                         facecolor=MID, edgecolor=LIGHT,
                         linewidth=0.8, clip_on=False)
    ax_cards.add_patch(rect)
    ax_cards.text(cx + cw / 2, cy + ch * 0.62, value,
                  transform=ax_cards.transAxes,
                  ha="center", va="center",
                  fontsize=11, fontweight="bold", color=color)
    ax_cards.text(cx + cw / 2, cy + ch * 0.22, label,
                  transform=ax_cards.transAxes,
                  ha="center", va="center",
                  fontsize=7.5, color=DIM)

# ------------------------------------------
# Row 1 col 0: NAV curve
# ------------------------------------------

ax1 = fig.add_subplot(gs[1, 0])
style_ax(ax1, "NAV")
ax1.plot(plot_dates, portfolio, color=GREEN, linewidth=1.2)
ax1.axhline(1.0, color=DIM, linewidth=0.6, linestyle="--")
ax1.fill_between(plot_dates, portfolio, 1.0,
                 where=[p >= 1 for p in portfolio],
                 alpha=0.12, color=GREEN)
ax1.fill_between(plot_dates, portfolio, 1.0,
                 where=[p < 1 for p in portfolio],
                 alpha=0.12, color=RED)
ax1.yaxis.set_major_formatter(pct_fmt)
ax1.tick_params(axis="x", rotation=30)

# ------------------------------------------
# Row 1 col 1: Drawdown
# ------------------------------------------

ax2 = fig.add_subplot(gs[1, 1])
style_ax(ax2, "Drawdown")
ax2.fill_between(plot_dates, dd_series, 0, alpha=0.45, color=RED)
ax2.plot(plot_dates, dd_series, color=RED, linewidth=0.9)
ax2.axhline(0, color=DIM, linewidth=0.5)
ax2.yaxis.set_major_formatter(pct2_fmt)
ax2.tick_params(axis="x", rotation=30)

# ------------------------------------------
# Row 1 col 2: Rolling 21d Sharpe
# ------------------------------------------

ax3 = fig.add_subplot(gs[1, 2])
style_ax(ax3, "Rolling 21-Day Sharpe")
rolling_sharpe = (
                         r.rolling(21).mean() - daily_rf
                 ) / r.rolling(21).std() * np.sqrt(252)
ax3.plot(r.index, rolling_sharpe, color=BLUE, linewidth=1.0)
ax3.axhline(0, color=DIM, linewidth=0.5, linestyle="--")
ax3.axhline(1, color=GREEN, linewidth=0.4, linestyle=":")
ax3.axhline(-1, color=RED, linewidth=0.4, linestyle=":")
ax3.tick_params(axis="x", rotation=30)

# ------------------------------------------
# Row 2 col 0: Returns distribution
# ------------------------------------------

ax4 = fig.add_subplot(gs[2, 0])
style_ax(ax4, "Daily Returns Distribution")

n_bins = 25
counts, bin_edges, patches = ax4.hist(
    r * 100, bins=n_bins, edgecolor="none", linewidth=0
)
for patch, left in zip(patches, bin_edges[:-1]):
    patch.set_facecolor(RED if left < 0 else GREEN)
    patch.set_alpha(0.7)

# Normal overlay
mu, sigma = r.mean() * 100, r.std() * 100
x_range = np.linspace(bin_edges[0], bin_edges[-1], 200)
bw = bin_edges[1] - bin_edges[0]
normal_y = stats.norm.pdf(x_range, mu, sigma) * len(r) * bw
ax4.plot(x_range, normal_y, color=AMBER, linewidth=1.2, linestyle="--", label="Normal fit")

ax4.axvline(st['var_95'] * 100, color=RED, linewidth=1.0, linestyle="--",
            label=f"VaR 95% {st['var_95']:.2%}")
ax4.axvline(st['cvar_95'] * 100, color=RED, linewidth=0.7, linestyle=":",
            label=f"CVaR 95% {st['cvar_95']:.2%}")
ax4.set_xlabel("Daily Return (%)", color=DIM, fontsize=8)
ax4.legend(fontsize=7, facecolor=MID, edgecolor=LIGHT, labelcolor=TEXT)

# ------------------------------------------
# Row 2 col 1: QQ plot
# ------------------------------------------

ax5 = fig.add_subplot(gs[2, 1])
style_ax(ax5, "QQ Plot (vs Normal)")
(osm, osr), (slope, intercept, _) = stats.probplot(r, dist="norm")
ax5.scatter(osm, osr, color=BLUE, s=10, alpha=0.7)
line_x = np.array([osm[0], osm[-1]])
ax5.plot(line_x, slope * line_x + intercept, color=AMBER,
         linewidth=1.0, linestyle="--")
ax5.set_xlabel("Theoretical quantiles", color=DIM, fontsize=8)
ax5.set_ylabel("Sample quantiles", color=DIM, fontsize=8)

# ------------------------------------------
# Row 2 col 2: Rolling volatility
# ------------------------------------------

ax6 = fig.add_subplot(gs[2, 2])
style_ax(ax6, "Rolling 21-Day Volatility (Ann.)")
rolling_vol = r.rolling(21).std() * np.sqrt(252)
ax6.plot(r.index, rolling_vol, color=AMBER, linewidth=1.0)
ax6.axhline(st['annual_vol'], color=DIM, linewidth=0.6,
            linestyle="--", label=f"Full-period avg {st['annual_vol']:.1%}")
ax6.yaxis.set_major_formatter(pct2_fmt)
ax6.tick_params(axis="x", rotation=30)
ax6.legend(fontsize=7, facecolor=MID, edgecolor=LIGHT, labelcolor=TEXT)

# ------------------------------------------
# Row 3 col 0: Per-ticker contribution
# ------------------------------------------

ax7 = fig.add_subplot(gs[3, 0])
style_ax(ax7, "Ticker Contribution to Return")
tc = st['ticker_contribs']
names = list(tc.keys())
vals = [tc[t] * 100 for t in names]
colors = [GREEN if v >= 0 else RED for v in vals]
y_pos = range(len(names))
bars = ax7.barh(y_pos, vals, color=colors, alpha=0.75, height=0.55)
ax7.set_yticks(list(y_pos))
ax7.set_yticklabels(names, color=TEXT, fontsize=8)
ax7.axvline(0, color=DIM, linewidth=0.6)
ax7.set_xlabel("Contribution (%)", color=DIM, fontsize=8)
for bar, v in zip(bars, vals):
    ax7.text(v + (0.05 if v >= 0 else -0.05), bar.get_y() + bar.get_height() / 2,
             f"{v:+.2f}%", va="center",
             ha="left" if v >= 0 else "right",
             color=TEXT, fontsize=7.5)

# ------------------------------------------
# Row 3 col 1: Win/loss profile
# ------------------------------------------

ax8 = fig.add_subplot(gs[3, 1])
style_ax(ax8, "Win / Loss Profile")
ax8.set_xticks([])
ax8.set_yticks([])
ax8.grid(False)

profile_items = [
    ("Win Rate", f"{st['win_rate']:.1%}", GREEN),
    ("Avg Win", f"{st['avg_win']:.2%}", GREEN),
    ("Avg Loss", f"{st['avg_loss']:.2%}", RED),
    ("Win/Loss Ratio", f"{st['win_loss_ratio']:.3f}", GREEN if st['win_loss_ratio'] >= 1 else RED),
    ("Best Day", f"{st['best_day']:+.2%}", GREEN),
    ("Worst Day", f"{st['worst_day']:+.2%}", RED),
    ("Best Week", f"{st['best_week']:+.2%}", GREEN),
    ("Worst Week", f"{st['worst_week']:+.2%}", RED),
    ("Max Win Streak", f"{st['max_win_streak']}d", GREEN),
    ("Max Lose Streak", f"{st['max_lose_streak']}d", RED),
]

row_h = 1.0 / (len(profile_items) + 1)
for i, (label, value, col) in enumerate(profile_items):
    y = 1.0 - (i + 1) * row_h
    ax8.text(0.05, y + row_h * 0.3, label,
             transform=ax8.transAxes,
             ha="left", va="center",
             fontsize=8, color=DIM)
    ax8.text(0.95, y + row_h * 0.3, value,
             transform=ax8.transAxes,
             ha="right", va="center",
             fontsize=8.5, fontweight="bold", color=col)
    if i < len(profile_items) - 1:
        ax8.plot([0.03, 0.97], [y, y],
                 color=LIGHT, linewidth=0.4,
                 transform=ax8.transAxes, clip_on=False)

# ------------------------------------------
# Row 3 col 2: Distribution stats table
# ------------------------------------------

ax9 = fig.add_subplot(gs[3, 2])
style_ax(ax9, "Distribution & Tail Risk")
ax9.set_xticks([])
ax9.set_yticks([])
ax9.grid(False)

jb_sig_str = f"yes (p={st['jb_p']:.3f})" if st['jb_p'] < 0.05 else f"no (p={st['jb_p']:.3f})"

dist_items = [
    ("Skewness", f"{st['skewness']:+.3f}", RED if st['skewness'] < -0.5 else AMBER if st['skewness'] < 0 else GREEN),
    ("Excess Kurtosis", f"{st['kurtosis']:+.3f}", RED if st['kurtosis'] > 2 else AMBER),
    ("Non-normal?", jb_sig_str, RED if st['jb_p'] < 0.05 else GREEN),
    ("VaR 95%", f"{st['var_95']:.2%}", RED),
    ("CVaR 95%", f"{st['cvar_95']:.2%}", RED),
    ("VaR 99%", f"{st['var_99']:.2%}", RED),
    ("CVaR 99%", f"{st['cvar_99']:.2%}", RED),
    ("CVaR/VaR ratio", f"{st['tail_ratio']:.3f}", RED if st['tail_ratio'] > 1.5 else AMBER),
    ("Beta (vs URTH)", f"{st['beta']:.3f}", BLUE),
    ("Alpha (vs URTH)", f"{st['alpha_ann']:+.2%}", GREEN if st['alpha_ann'] >= 0 else RED),
    ("R-squared", f"{st['r_squared']:.3f}", AMBER),
    ("Ulcer Index", f"{st['ulcer_index']:.4f}", AMBER),
    ("Avg DD Duration", f"{st['avg_dd_duration']:.1f}d", AMBER),
]

row_h = 1.0 / (len(dist_items) + 1)
for i, (label, value, col) in enumerate(dist_items):
    y = 1.0 - (i + 1) * row_h
    ax9.text(0.05, y + row_h * 0.3, label,
             transform=ax9.transAxes,
             ha="left", va="center",
             fontsize=8, color=DIM)
    ax9.text(0.95, y + row_h * 0.3, value,
             transform=ax9.transAxes,
             ha="right", va="center",
             fontsize=8.5, fontweight="bold", color=col)
    if i < len(dist_items) - 1:
        ax9.plot([0.03, 0.97], [y, y],
                 color=LIGHT, linewidth=0.4,
                 transform=ax9.transAxes, clip_on=False)

plt.savefig("portfolio_dashboard.png", dpi=150, bbox_inches="tight",
            facecolor=DARK)
plt.show()
print("Dashboard saved to portfolio_dashboard.png")