import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
import yfinance as yf
warnings.filterwarnings("ignore")

tickers       = ["SPY", "GLD", "QQQ", "EEM", "IWM"]
start         = "2015-01-01"
end           = "2024-01-01"

lookback      = 252
rebal_freq    = 21
risk_aversion = 2.5
tau           = 0.05
max_weight    = 0.40
min_weight    = 0.02
view_lookback = 63
view_conf     = 0.0001
risk_free     = 0.04 / 252


print("Downloading data from Yahoo Finance...")
raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
prices = raw["Close"].dropna()
print(f"Loaded {len(prices)} rows for {list(prices.columns)}")

print("Fetching market caps via fast_info (no rate-limit risk)...")
mcap_arr = np.ones(len(prices.columns))
for j, t in enumerate(prices.columns):
    try:
        fi     = yf.Ticker(t).fast_info
        shares = getattr(fi, "shares", None)
        price  = getattr(fi, "last_price", None)
        if shares and price:
            mcap_arr[j] = shares * price
        else:
            mcap_arr[j] = prices[t].iloc[-1]   # fallback: use latest price as proxy
    except Exception:
        mcap_arr[j] = prices[t].iloc[-1]        # fallback: use latest price as proxy
w_mkt_real = mcap_arr / mcap_arr.sum()
print(f"Market-cap weights: { {t: f'{w:.1%}' for t, w in zip(prices.columns, w_mkt_real)} }")


def ledoit_wolf_cov(returns: np.ndarray) -> np.ndarray:
    lw = LedoitWolf().fit(returns)
    return lw.covariance_ * 252


def equilibrium_returns(sigma: np.ndarray, w_mkt: np.ndarray,
                        lam: float = risk_aversion) -> np.ndarray:
    return lam * sigma @ w_mkt


def build_views_momentum(returns_window: pd.DataFrame):
    n = returns_window.shape[1]
    cum_ret = (1 + returns_window).prod() - 1
    ranked  = cum_ret.rank()
    top     = ranked[ranked > n / 2].index.tolist()
    bot     = ranked[ranked <= n / 2].index.tolist()
    assets  = returns_window.columns.tolist()

    views, q_vals = [], []

    p_row = np.zeros(n)
    for t in top:  p_row[assets.index(t)] =  1 / len(top)
    for b in bot:  p_row[assets.index(b)] = -1 / len(bot)
    views.append(p_row)
    q_vals.append(0.04)

    p = np.array(views)
    q = np.array(q_vals)
    k = len(views)
    omega = np.diag([view_conf] * k)
    return p, q, omega


def black_litterman(sigma: np.ndarray, pi: np.ndarray,
                    p: np.ndarray, q: np.ndarray,
                    omega: np.ndarray, tau_: float = tau):
    tau_sigma_inv = np.linalg.inv(tau_ * sigma)
    omega_inv     = np.linalg.inv(omega)

    a = tau_sigma_inv + p.T @ omega_inv @ p
    b = tau_sigma_inv @ pi + p.T @ omega_inv @ q

    a_inv  = np.linalg.inv(a)
    mu_bl  = a_inv @ b
    cov_bl = sigma + a_inv
    return mu_bl, cov_bl


def optimise_portfolio(mu: np.ndarray, sigma: np.ndarray,
                       lam: float = risk_aversion,
                       w_min: float = min_weight,
                       w_max: float = max_weight) -> np.ndarray:
    n  = len(mu)
    w0 = np.ones(n) / n

    def neg_utility(w):
        return -(mu @ w - lam / 2 * w @ sigma @ w)

    def neg_utility_grad(w):
        return -(mu - lam * sigma @ w)

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds = [(w_min, w_max)] * n

    res = minimize(neg_utility, w0, jac=neg_utility_grad,
                   method="SLSQP", bounds=bounds,
                   constraints=constraints,
                   options={"ftol": 1e-12, "maxiter": 500})

    if res.success:
        return res.x
    return np.ones(n) / n


def min_variance_portfolio(sigma: np.ndarray,
                           w_min: float = min_weight,
                           w_max: float = max_weight) -> np.ndarray:
    n  = sigma.shape[0]
    w0 = np.ones(n) / n

    def port_var(w): return w @ sigma @ w

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds = [(w_min, w_max)] * n
    res = minimize(port_var, w0, method="SLSQP",
                   bounds=bounds, constraints=constraints,
                   options={"ftol": 1e-12, "maxiter": 500})
    return res.x if res.success else np.ones(n) / n


def run_backtest(prices: pd.DataFrame, w_mkt_fixed: np.ndarray = None):
    returns  = prices.pct_change().dropna()
    n_assets = len(prices.columns)
    dates    = returns.index

    port_returns = {"BL": [], "EqualWeight": [], "MarketCap": [], "MinVar": []}
    port_weights = {"BL": [], "EqualWeight": [], "MarketCap": [], "MinVar": []}
    rebal_dates  = []
    bl_details   = []

    w = {k: np.ones(n_assets) / n_assets for k in port_returns}

    # precompute return matrix for vectorised daily recording
    ret_matrix = returns.values  # shape (T, n_assets)

    print(f"\nRunning backtest: {dates[lookback].date()} → {dates[-1].date()}")
    print(f"Assets: {list(prices.columns)}\n")

    rebal_indices = range(lookback, len(dates))

    for i in rebal_indices:
        date      = dates[i]
        ret_today = ret_matrix[i]

        for k in port_returns:
            port_returns[k].append(w[k] @ ret_today)

        if (i - lookback) % rebal_freq == 0:
            window_ret  = ret_matrix[i - lookback: i]
            view_window = returns.iloc[i - view_lookback: i]

            sigma = ledoit_wolf_cov(window_ret)
            w_mkt = w_mkt_fixed if w_mkt_fixed is not None else np.ones(n_assets) / n_assets

            pi = equilibrium_returns(sigma, w_mkt)

            p_mat, q_vec, omega = build_views_momentum(view_window)

            mu_bl, cov_bl = black_litterman(sigma, pi, p_mat, q_vec, omega)

            w["BL"]          = optimise_portfolio(mu_bl, cov_bl)
            w["EqualWeight"] = np.ones(n_assets) / n_assets
            w["MarketCap"]   = w_mkt.copy()
            w["MinVar"]      = min_variance_portfolio(sigma)

            rebal_dates.append(date)
            bl_details.append({
                "date":       date,
                "pi":         pi.copy(),
                "mu_bl":      mu_bl.copy(),
                "weights_bl": w["BL"].copy()
            })

            for k in port_returns:
                port_weights[k].append(w[k].copy())

    idx    = dates[lookback:]
    ret_df = pd.DataFrame(port_returns, index=idx)
    return ret_df, pd.DataFrame(bl_details).set_index("date"), prices.columns.tolist()


def performance_stats(ret_series: pd.Series, rf: float = risk_free) -> dict:
    excess   = ret_series - rf
    ann_ret  = ret_series.mean() * 252
    ann_vol  = ret_series.std() * np.sqrt(252)
    sharpe   = excess.mean() / ret_series.std() * np.sqrt(252) if ret_series.std() > 0 else 0

    cum      = (1 + ret_series).cumprod()
    roll_max = cum.cummax()
    dd       = (cum - roll_max) / roll_max
    max_dd   = dd.min()
    calmar   = ann_ret / abs(max_dd) if max_dd != 0 else 0

    hit_rate = (ret_series > 0).mean()

    return {
        "Ann. Return":     f"{ann_ret:.2%}",
        "Ann. Volatility": f"{ann_vol:.2%}",
        "Sharpe Ratio":    f"{sharpe:.2f}",
        "Max Drawdown":    f"{max_dd:.2%}",
        "Calmar Ratio":    f"{calmar:.2f}",
        "Hit Rate":        f"{hit_rate:.2%}",
    }


def print_stats(ret_df: pd.DataFrame):
    print("=" * 65)
    print(f"{'Metric':<22}", end="")
    for col in ret_df.columns:
        print(f"{col:>10}", end="")
    print()
    print("-" * 65)

    stats = {col: performance_stats(ret_df[col]) for col in ret_df.columns}
    keys  = list(next(iter(stats.values())).keys())
    for k in keys:
        print(f"{k:<22}", end="")
        for col in ret_df.columns:
            print(f"{stats[col][k]:>10}", end="")
        print()
    print("=" * 65)


def plot_results(ret_df: pd.DataFrame, bl_details: pd.DataFrame, asset_names: list):
    cum = (1 + ret_df).cumprod()

    fig = plt.figure(figsize=(16, 14))
    fig.patch.set_facecolor("#0f0f1a")
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.4, wspace=0.35)

    colors = {"BL": "#00d4ff", "EqualWeight": "#ff6b6b",
              "MarketCap": "#ffd93d", "MinVar": "#6bcb77"}

    ax1 = fig.add_subplot(gs[0, :])
    ax1.set_facecolor("#1a1a2e")
    for col in cum.columns:
        ax1.plot(cum.index, cum[col], label=col, color=colors[col], lw=1.8)
    ax1.set_title("Cumulative Returns", color="white", fontsize=13, pad=10)
    ax1.legend(framealpha=0, labelcolor="white", fontsize=10)
    ax1.tick_params(colors="white")
    ax1.set_ylabel("Portfolio Value (start=1)", color="white")
    for spine in ax1.spines.values(): spine.set_color("#333355")
    ax1.yaxis.label.set_color("white")

    ax2 = fig.add_subplot(gs[1, 0])
    ax2.set_facecolor("#1a1a2e")
    for col in cum.columns:
        dd = (cum[col] / cum[col].cummax() - 1)
        ax2.fill_between(dd.index, dd, alpha=0.4, color=colors[col], label=col)
    ax2.set_title("Drawdowns", color="white", fontsize=11, pad=8)
    ax2.tick_params(colors="white")
    ax2.set_ylabel("Drawdown", color="white")
    ax2.legend(framealpha=0, labelcolor="white", fontsize=8)
    for spine in ax2.spines.values(): spine.set_color("#333355")

    ax3 = fig.add_subplot(gs[1, 1])
    ax3.set_facecolor("#1a1a2e")
    for col in ["BL", "EqualWeight"]:
        rs = ret_df[col].rolling(63).apply(
            lambda x: (x.mean() / x.std() * np.sqrt(252)) if x.std() > 0 else 0
        )
        ax3.plot(rs.index, rs, label=col, color=colors[col], lw=1.5)
    ax3.axhline(0, color="white", lw=0.5, linestyle="--")
    ax3.set_title("Rolling 63-day Sharpe (BL vs EW)", color="white", fontsize=11, pad=8)
    ax3.tick_params(colors="white")
    ax3.set_ylabel("Sharpe", color="white")
    ax3.legend(framealpha=0, labelcolor="white", fontsize=8)
    for spine in ax3.spines.values(): spine.set_color("#333355")

    ax4 = fig.add_subplot(gs[2, 0])
    ax4.set_facecolor("#1a1a2e")
    if "weights_bl" in bl_details.columns:
        wt_matrix = np.vstack(bl_details["weights_bl"].values)
        palette   = plt.cm.plasma(np.linspace(0.1, 0.9, len(asset_names)))
        bottom    = np.zeros(len(bl_details))
        for j, asset in enumerate(asset_names):
            ax4.bar(bl_details.index, wt_matrix[:, j], bottom=bottom,
                    label=asset, color=palette[j], width=20)
            bottom += wt_matrix[:, j]
    ax4.set_title("BL Portfolio Weights Over Time", color="white", fontsize=11, pad=8)
    ax4.tick_params(colors="white")
    ax4.set_ylabel("Weight", color="white")
    ax4.legend(framealpha=0, labelcolor="white", fontsize=7, loc="upper left")
    for spine in ax4.spines.values(): spine.set_color("#333355")

    ax5 = fig.add_subplot(gs[2, 1])
    ax5.set_facecolor("#1a1a2e")
    if len(bl_details) > 0:
        latest = bl_details.iloc[-1]
        x      = np.arange(len(asset_names))
        w_bar  = 0.35
        ax5.bar(x - w_bar/2, latest["pi"] * 100,    w_bar, label="Equilibrium Π",
                color="#00d4ff", alpha=0.8)
        ax5.bar(x + w_bar/2, latest["mu_bl"] * 100, w_bar, label="BL Posterior μ",
                color="#ff6b6b", alpha=0.8)
        ax5.set_xticks(x)
        ax5.set_xticklabels(asset_names, color="white", fontsize=9)
        ax5.axhline(0, color="white", lw=0.5)
    ax5.set_title("Latest: Equilibrium vs BL Returns (%)", color="white", fontsize=11, pad=8)
    ax5.tick_params(colors="white")
    ax5.set_ylabel("Ann. Return (%)", color="white")
    ax5.legend(framealpha=0, labelcolor="white", fontsize=8)
    for spine in ax5.spines.values(): spine.set_color("#333355")

    fig.suptitle("Black-Litterman Backtest", color="white",
                 fontsize=16, fontweight="bold", y=0.98)
    plt.show()


ret_df, bl_details, asset_names = run_backtest(prices, w_mkt_fixed=w_mkt_real)

print("\n── Performance Summary ──")
print_stats(ret_df)

print("\n── Latest BL rebalance ──")
if len(bl_details) > 0:
    latest = bl_details.iloc[-1]
    print(f"\nDate: {latest.name.date()}")
    print(f"\n{'Asset':<10} {'Equil Π':>10} {'BL μ':>10} {'Weight':>10}")
    print("-" * 42)
    for i, a in enumerate(asset_names):
        print(f"{a:<10} {latest['pi'][i]*100:>9.2f}% "
              f"{latest['mu_bl'][i]*100:>9.2f}% "
              f"{latest['weights_bl'][i]*100:>9.2f}%")

plot_results(ret_df, bl_details, asset_names)