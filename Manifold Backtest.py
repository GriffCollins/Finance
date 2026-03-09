import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")

try:
    import yfinance as yf
    use_real_data = True
except ImportError:
    use_real_data = False

# CONFIG
tickers = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "AMD", "INTC", "CRM",
    "BAC",  "GS",   "MS",    "BLK",  "WFC",  "C",   "USB",  "TFC",
    "JNJ",  "UNH",  "PFE",  "LLY",  "TMO",  "CVS",
    "CAT",  "BA",   "HON",  "RTX",  "DE",  "MMM",  "FDX",
    "KO",   "PG",   "MCD",   "NKE",  "XOM",  "CVX",  "COP", "NEE", "AMT",
]
start         = "2018-01-01"
end           = "2023-01-01"
lookback      = 252
rebal_freq    = 21
risk_aversion = 2.5
tau           = 0.05
max_weight    = 0.10
min_weight    = 0.00
risk_free     = 0.04 / 252
stress_pctile = 75


# Data

def load_data():
    if use_real_data:
        print("Downloading data from Yahoo Finance...")
        raw    = yf.download(tickers, start=start, end=end,
                             auto_adjust=True, progress=False)
        prices = raw["Close"].dropna()
        prices = prices.loc[:, prices.isnull().mean() < 0.01].dropna()
        print(f"Loaded {len(prices)} rows, {prices.shape[1]} assets")
        return prices
    else:
        print("Generating synthetic data for", len(tickers), "assets")
        np.random.seed(42)
        n, t  = len(tickers), 2270
        dates = pd.bdate_range(start=start, periods=t)

        # Realistic sector structure: 5 sectors of 10 stocks each
        # Within-sector correlation 0.65, cross-sector 0.25
        n_sectors = 5
        stocks_per = n // n_sectors
        corr = np.full((n, n), 0.25)
        for s in range(n_sectors):
            idx = slice(s*stocks_per, (s+1)*stocks_per)
            corr[idx, idx] = 0.65
        np.fill_diagonal(corr, 1.0)

        # Heterogeneous vols: tech=35%, financials=22%, healthcare=18%, industrials=20%, other=16%
        sector_vols = [0.35, 0.22, 0.18, 0.20, 0.16]
        vols = np.concatenate([
            np.random.uniform(sv*0.7, sv*1.3, stocks_per) / np.sqrt(252)
            for sv in sector_vols
        ])

        # Heterogeneous drifts
        sector_drifts = [0.14, 0.10, 0.09, 0.08, 0.07]
        drift = np.concatenate([
            np.random.uniform(sd*0.7, sd*1.3, stocks_per) / 252
            for sd in sector_drifts
        ])

        l     = np.linalg.cholesky(corr)
        z     = np.random.randn(t, n) @ l.T
        rets  = drift + z * vols
        prices = pd.DataFrame(100 * np.cumprod(1 + rets, axis=0),
                               index=dates, columns=tickers)
        return prices


# Riemannian Geometry

def spd_sqrt(a):
    vals, vecs = np.linalg.eigh(a)
    vals = np.maximum(vals, 1e-10)
    s    = vecs @ np.diag(np.sqrt(vals))   @ vecs.T
    si   = vecs @ np.diag(1/np.sqrt(vals)) @ vecs.T
    return s, si

def spd_log(a):
    vals, vecs = np.linalg.eigh(a)
    vals = np.maximum(vals, 1e-10)
    return vecs @ np.diag(np.log(vals)) @ vecs.T

def spd_exp(a):
    vals, vecs = np.linalg.eigh(a)
    return vecs @ np.diag(np.exp(vals)) @ vecs.T

def riemannian_distance(a, b):
    _, ai = spd_sqrt(a)
    m     = ai @ b @ ai
    logm  = spd_log(m)
    return float(np.sqrt(np.maximum(np.einsum('ij,ij->', logm, logm), 0)))

def geodesic_point(a, b, t):
    s, si = spd_sqrt(a)
    m     = si @ b @ si
    vals, vecs = np.linalg.eigh(m)
    vals  = np.maximum(vals, 1e-10)
    mt    = vecs @ np.diag(vals ** t) @ vecs.T
    return s @ mt @ s

def frechet_mean(mats, max_iter=30, tol=1e-6):
    # log-Euclidean warm start — precompute all logs once before iteration
    log_mats = [spd_log(m) for m in mats]
    s = spd_exp(np.mean(log_mats, axis=0))
    for _ in range(max_iter):
        s_s, s_si = spd_sqrt(s)
        grad = np.mean([spd_log(s_si @ m @ s_si) for m in mats], axis=0)
        s_new = s_s @ spd_exp(grad) @ s_s
        if np.linalg.norm(grad, 'fro') < tol:
            break
        s = s_new
    return s

def riemannian_covariance(returns, n_blocks=4):
    """
    Estimate covariance via Fréchet mean of sub-period LW covariances.
    For n assets we need each block to have > n observations.
    With 252 days and 50 assets: 4 blocks of 63 days each is borderline
    but LW handles it. The Fréchet mean then averages across time periods
    on the manifold rather than in flat space.
    """
    t, n     = returns.shape
    # ensure each block has enough observations for LW
    min_obs  = n + 10
    n_blocks = min(n_blocks, t // min_obs)
    n_blocks = max(n_blocks, 2)
    bsize    = t // n_blocks
    cov_list = []
    for k in range(n_blocks):
        blk = returns[k*bsize:(k+1)*bsize]
        if blk.shape[0] < min_obs:
            continue
        lw = LedoitWolf().fit(blk)
        c  = lw.covariance_ * 252 + np.eye(n) * 1e-6
        cov_list.append(c)
    if len(cov_list) < 2:
        lw = LedoitWolf().fit(returns)
        return lw.covariance_ * 252 + np.eye(n) * 1e-6
    return frechet_mean(cov_list)


# Regime Detector

class RegimeDetector:
    def __init__(self, history_len=20):
        self.history_len  = history_len
        self.cov_history  = []
        self.dist_history = []
        self.mean_cov     = None

    def update(self, sigma):
        self.cov_history.append(sigma)
        if len(self.cov_history) > self.history_len:
            self.cov_history.pop(0)
        if len(self.cov_history) >= 4:
            self.mean_cov = frechet_mean(self.cov_history)
            self.dist_history.append(riemannian_distance(sigma, self.mean_cov))

    def stress_score(self):
        if len(self.dist_history) < 3:
            return 0.0
        pct = np.percentile(self.dist_history, stress_pctile)
        return float(np.clip((self.dist_history[-1] - pct) / (pct + 1e-9), 0, 1))

    def is_stress(self):
        return self.stress_score() > 0.3


# Optimisers

def _solve_min_var(sigma, w_min, w_max, n_restarts=3):
    """
    Minimum variance with multiple random restarts.
    Falls back to inverse-vol if all restarts fail.
    """
    n   = sigma.shape[0]
    eq  = {"type": "eq", "fun": lambda w: w.sum()-1, "jac": lambda w: np.ones(n)}
    bds = [(w_min, w_max)] * n
    best_w, best_val = None, np.inf

    # warm-start first restart from inverse-vol (saves ~1 iteration vs pure random)
    iv = 1.0 / np.sqrt(np.diag(sigma))
    w0_iv = np.clip(iv / iv.sum(), w_min, w_max)
    w0_iv /= w0_iv.sum()
    restarts = [w0_iv] + [None] * (n_restarts - 1)

    for r in restarts:
        if r is None:
            w0 = np.random.dirichlet(np.ones(n))
            w0 = np.clip(w0, w_min, w_max); w0 /= w0.sum()
        else:
            w0 = r
        res = minimize(lambda w: w @ sigma @ w,
                       w0, jac=lambda w: 2*sigma@w,
                       method="SLSQP", bounds=bds, constraints=[eq],
                       options={"ftol": 1e-10, "maxiter": 1000})
        if res.success and res.fun < best_val:
            best_val, best_w = res.fun, res.x.copy()

    if best_w is None:
        iv = 1.0 / np.sqrt(np.diag(sigma))
        best_w = iv / iv.sum()

    best_w = np.clip(best_w, 0, None)
    return best_w / best_w.sum()


def euclidean_min_variance(sigma, w_min=None, w_max=None):
    w_min = min_weight if w_min is None else w_min
    w_max = max_weight if w_max is None else w_max
    return _solve_min_var(sigma, w_min, w_max)


def riemannian_min_variance(sigma, w_min=None, w_max=None):
    """
    Riemannian minimum variance: minimise w'Σw but using the Riemannian
    covariance (Fréchet mean) as input. The objective is standard portfolio
    variance — the Riemannian geometry enters through the covariance estimate,
    not the objective function itself (which would be numerically intractable
    for 50 assets). This is the correct and stable formulation.
    """
    w_min = min_weight if w_min is None else w_min
    w_max = max_weight if w_max is None else w_max
    return _solve_min_var(sigma, w_min, w_max)


def _solve_bl(sigma, returns_window, asset_names, w_min, w_max):
    """Shared BL solve used by both riemannian_bl and euclidean_bl."""
    n   = sigma.shape[0]
    ew  = np.ones(n) / n
    pi  = risk_aversion * sigma @ ew

    cum_ret = pd.Series(
        (1 + pd.DataFrame(returns_window, columns=asset_names)).prod().values - 1,
        index=asset_names)
    ranked  = cum_ret.rank()
    top_idx = [i for i, a in enumerate(asset_names) if ranked[a] > n*0.8]
    bot_idx = [i for i, a in enumerate(asset_names) if ranked[a] <= n*0.2]

    if not top_idx or not bot_idx:
        return _solve_min_var(sigma, w_min, w_max)

    p = np.zeros((1, n))
    for i in top_idx: p[0, i] =  1/len(top_idx)
    for i in bot_idx: p[0, i] = -1/len(bot_idx)
    q     = np.array([0.04])
    omega = np.diag([0.0002])

    try:
        tsi    = np.linalg.inv(tau * sigma)
        oi     = np.linalg.inv(omega)
        a      = tsi + p.T @ oi @ p
        b      = tsi @ pi + p.T @ oi @ q
        ai     = np.linalg.inv(a)
        mu_bl  = ai @ b
        cov_bl = sigma + ai + np.eye(n)*1e-6
    except np.linalg.LinAlgError:
        return _solve_min_var(sigma, w_min, w_max)

    eq  = {"type": "eq", "fun": lambda w: w.sum()-1, "jac": lambda w: np.ones(n)}
    bds = [(w_min, w_max)] * n
    best_w, best_val = None, np.inf
    for _ in range(3):
        w0  = np.random.dirichlet(np.ones(n))
        w0  = np.clip(w0, w_min, w_max); w0 /= w0.sum()
        res = minimize(lambda w: -(mu_bl@w - risk_aversion/2*w@cov_bl@w),
                       w0, jac=lambda w: -(mu_bl - risk_aversion*cov_bl@w),
                       method="SLSQP", bounds=bds, constraints=[eq],
                       options={"ftol": 1e-10, "maxiter": 1000})
        if res.success and res.fun < best_val:
            best_val, best_w = res.fun, res.x.copy()

    if best_w is None: best_w = ew
    best_w = np.clip(best_w, 0, None)
    return best_w / best_w.sum()


def euclidean_bl(sigma, returns_window, asset_names,
                 w_min=None, w_max=None):
    """Standard flat-space BL — identical logic to riemannian_bl but uses
    Ledoit-Wolf Euclidean covariance instead of Riemannian Fréchet mean."""
    w_min = min_weight if w_min is None else w_min
    w_max = max_weight if w_max is None else w_max
    return _solve_bl(sigma, returns_window, asset_names, w_min, w_max)


def riemannian_bl(sigma, returns_window, asset_names,
                  w_min=None, w_max=None):
    """BL with Riemannian covariance. Momentum view: top quintile vs bottom quintile."""
    w_min = min_weight if w_min is None else w_min
    w_max = max_weight if w_max is None else w_max
    return _solve_bl(sigma, returns_window, asset_names, w_min, w_max)


def regime_aware_portfolio(sigma_calm, sigma_stress, stress_score,
                            w_min=None, w_max=None):
    w_min = min_weight if w_min is None else w_min
    w_max = max_weight if w_max is None else w_max
    t = float(np.clip(stress_score, 0, 1))
    try:
        sb = geodesic_point(sigma_calm, sigma_stress, t)
        sb += np.eye(sb.shape[0]) * 1e-6
    except Exception:
        sb = (1-t)*sigma_calm + t*sigma_stress
    return _solve_min_var(sb, w_min, w_max)


# Backtest Engine

def run_backtest(prices):
    returns  = prices.pct_change().dropna()
    n        = prices.shape[1]
    dates    = returns.index
    names    = list(prices.columns)

    # in run_backtest, inside the rebalance block, add to strategies list:
    strategies = ["RiemMinVar", "RiemBL", "RegimeAware", "EuclidMinVar", "FlatBL", "EqualWeight"]
    port_returns = {s: [] for s in strategies}
    rebal_info   = []
    detector     = RegimeDetector(history_len=15)
    sigma_stress = np.eye(n) * 0.04

    # each strategy starts equal weight — independent arrays
    w = {s: np.ones(n)/n for s in strategies}

    # precompute return matrix for vectorised daily recording
    ret_matrix = returns.values

    print(f"Running  {dates[lookback].date()} → {dates[-1].date()}")
    print(f"{n} assets  |  lookback {lookback}d  |  rebal every {rebal_freq}d\n")

    n_rebal = 0
    for i in range(lookback, len(dates)):
        ret_today = ret_matrix[i]
        for s in strategies:
            port_returns[s].append(float(w[s] @ ret_today))

        if (i - lookback) % rebal_freq == 0:
            window = returns.iloc[i-lookback:i]

            sigma_riem = riemannian_covariance(window.values, n_blocks=6)
            lw         = LedoitWolf().fit(window.values)
            sigma_eucl = lw.covariance_ * 252 + np.eye(n)*1e-6

            detector.update(sigma_riem)
            stress = detector.stress_score()
            if detector.is_stress():
                sigma_stress = geodesic_point(sigma_stress, sigma_riem, 0.3)

            # compute each weight vector independently
            ew_new    = np.ones(n) / n
            eucl_new  = euclidean_min_variance(sigma_eucl)
            riem_new  = riemannian_min_variance(sigma_riem)
            bl_new    = riemannian_bl(sigma_riem, window.values, names)
            reg_new   = regime_aware_portfolio(sigma_riem, sigma_stress, stress)
            eucl_bl   = euclidean_bl(sigma_eucl, window.values, names)

            # verify they differ before assigning
            if n_rebal == 0:
                diff_cov = np.max(np.abs(sigma_riem - sigma_eucl))
                print(f"  Max element diff Σ_riem vs Σ_eucl: {diff_cov:.6f}")
                diffs = {
                    "EuclidMinVar": np.max(np.abs(eucl_new - ew_new)),
                    "RiemMinVar":   np.max(np.abs(riem_new - ew_new)),
                    "RiemBL":       np.max(np.abs(bl_new   - ew_new)),
                    "RegimeAware":  np.max(np.abs(reg_new  - ew_new)),
                }
                print("  Max weight deviation from EW at first rebal:")
                for k, v in diffs.items():
                    flag = "  ← SAME AS EW (fallback)" if v < 1e-4 else ""
                    print(f"    {k:<15} {v:.6f}{flag}")
                print(f"  Top 5 RiemMinVar:   {[f'{x:.3f}' for x in sorted(riem_new, reverse=True)[:5]]}")
                print(f"  Top 5 EuclidMinVar: {[f'{x:.3f}' for x in sorted(eucl_new, reverse=True)[:5]]}")
                print(f"  Top 5 RiemBL:       {[f'{x:.3f}' for x in sorted(bl_new,   reverse=True)[:5]]}")
                print()

            w["EqualWeight"]  = ew_new
            w["EuclidMinVar"] = eucl_new
            w["RiemMinVar"]   = riem_new
            w["RiemBL"]       = bl_new
            w["RegimeAware"]  = reg_new
            w["FlatBL"]       = eucl_bl

            eig_r = np.linalg.eigvalsh(sigma_riem)
            eig_e = np.linalg.eigvalsh(sigma_eucl)
            rebal_info.append({
                "date":       dates[i],
                "stress":     stress,
                "is_stress":  detector.is_stress(),
                "riem_dist":  detector.dist_history[-1] if detector.dist_history else 0,
                "cond_riem":  eig_r[-1] / max(eig_r[0], 1e-10),
                "cond_eucl":  eig_e[-1] / max(eig_e[0], 1e-10),
                "w_riem":     riem_new.copy(),
                "w_regime":   reg_new.copy(),
            })
            n_rebal += 1
            if n_rebal % 12 == 0:
                print(f"  {dates[i].date()}  stress={stress:.2f}  "
                      f"cond_riem={eig_r[-1]/max(eig_r[0],1e-10):.1f}  "
                      f"cond_eucl={eig_e[-1]/max(eig_e[0],1e-10):.1f}")

    ret_df = pd.DataFrame(port_returns, index=dates[lookback:])
    rb_df  = pd.DataFrame(rebal_info).set_index("date")
    return ret_df, rb_df, names


# Analytics

def annual_returns(ret_df):
    return (1 + ret_df).resample("YE").prod() - 1

def stats(s):
    ann_r = s.mean() * 252
    ann_v = s.std()  * np.sqrt(252)
    sr    = (s.mean()-risk_free)/s.std()*np.sqrt(252) if s.std()>0 else 0
    cum   = (1+s).cumprod()
    mdd   = (cum/cum.cummax()-1).min()
    return {"Ann Return": ann_r, "Volatility": ann_v, "Sharpe": sr,
            "Max DD": mdd, "Calmar": ann_r/abs(mdd) if mdd!=0 else 0,
            "Hit Rate": (s>0).mean()}

def print_summary(ret_df):
    df  = pd.DataFrame({c: stats(ret_df[c]) for c in ret_df.columns}).T
    fmt = {"Ann Return":"{:.2%}","Volatility":"{:.2%}","Sharpe":"{:.2f}",
           "Max DD":"{:.2%}","Calmar":"{:.2f}","Hit Rate":"{:.2%}"}
    out = df.copy()
    for col, f in fmt.items(): out[col] = df[col].map(f.format)
    print(out.to_string())

def print_annual(ret_df):
    ann = annual_returns(ret_df)
    ann.index = ann.index.year
    print(ann.map("{:.1%}".format).to_string())


# Dashboard

def plot_results(ret_df, rb_df, names):
    cum = (1+ret_df).cumprod()
    colors = {"RiemMinVar":"#00d4ff","RiemBL":"#bf5fff",
              "RegimeAware":"#00ff99","FlatBL": "#ff9f43","EuclidMinVar":"#ff6b6b","EqualWeight":"#ffd93d"}

    fig = plt.figure(figsize=(18,16))
    fig.patch.set_facecolor("#0a0a1a")
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    def style(ax, title):
        ax.set_facecolor("#12122a")
        ax.set_title(title, color="white", fontsize=10, pad=8)
        ax.tick_params(colors="#aaaacc", labelsize=8)
        for sp in ax.spines.values(): sp.set_color("#2a2a4a")
        ax.yaxis.label.set_color("#aaaacc")

    ax1 = fig.add_subplot(gs[0,:])
    for col in cum.columns:
        ax1.plot(cum.index, cum[col], label=col, color=colors[col], lw=1.8)
    if "is_stress" in rb_df:
        for d in rb_df[rb_df["is_stress"]].index:
            ax1.axvspan(d, d+pd.Timedelta(days=rebal_freq), alpha=0.08, color="red", lw=0)
    style(ax1, "Cumulative Returns  (red = stress regime)")
    ax1.legend(framealpha=0, labelcolor="white", fontsize=9)
    ax1.set_ylabel("Portfolio Value", color="#aaaacc")

    ax2 = fig.add_subplot(gs[1,0])
    for col in ret_df.columns:
        dd = (cum[col]/cum[col].cummax()-1)*100
        ax2.fill_between(dd.index, dd, alpha=0.4, color=colors[col], label=col)
    style(ax2, "Drawdowns (%)"); ax2.set_ylabel("Drawdown %", color="#aaaacc")
    ax2.legend(framealpha=0, labelcolor="white", fontsize=7)

    ax3 = fig.add_subplot(gs[1,1])
    if "riem_dist" in rb_df:
        ax3.plot(rb_df.index, rb_df["riem_dist"], color="#00d4ff", lw=1.5)
        ax3.fill_between(rb_df.index, rb_df["riem_dist"], alpha=0.2, color="#00d4ff")
        ax3b = ax3.twinx()
        ax3b.plot(rb_df.index, rb_df["stress"], color="#ff6b6b", lw=1.2, ls="--")
        ax3b.set_ylim(0,2); ax3b.tick_params(colors="#aaaacc", labelsize=8)
        ax3b.set_ylabel("Stress score", color="#ff6b6b")
    style(ax3, "Geodesic Distance (regime signal)")
    ax3.set_ylabel("Riemannian distance", color="#aaaacc")

    ax4 = fig.add_subplot(gs[1,2])
    if "cond_riem" in rb_df:
        ax4.plot(rb_df.index, rb_df["cond_riem"], color="#00ff99", lw=1.5, label="Riemannian")
        ax4.plot(rb_df.index, rb_df["cond_eucl"], color="#ff6b6b", lw=1.5, ls="--", label="Euclidean")
    style(ax4, "Condition Number (lower = better)")
    ax4.set_ylabel("Condition number", color="#aaaacc")
    ax4.legend(framealpha=0, labelcolor="white", fontsize=8)

    ax5 = fig.add_subplot(gs[2,0])
    for col in ret_df.columns:
        rs = ret_df[col].rolling(63).apply(
            lambda x: x.mean()/x.std()*np.sqrt(252) if x.std()>0 else 0)
        ax5.plot(rs.index, rs, color=colors[col], lw=1.3, label=col)
    ax5.axhline(0, color="white", lw=0.5, ls="--")
    style(ax5, "Rolling 63-day Sharpe")
    ax5.legend(framealpha=0, labelcolor="white", fontsize=7)

    ax6 = fig.add_subplot(gs[2,1])
    if "w_riem" in rb_df:
        wt  = np.vstack(rb_df["w_riem"].values)
        pal = plt.cm.tab20(np.linspace(0,1,min(20,len(names))))
        bot = np.zeros(len(rb_df))
        for j in range(min(20, len(names))):
            ax6.bar(rb_df.index, wt[:,j], bottom=bot, color=pal[j], width=20)
            bot += wt[:,j]
    style(ax6, "RiemMinVar Weights"); ax6.set_ylabel("Weight", color="#aaaacc")

    ax7 = fig.add_subplot(gs[2,2])
    ann = annual_returns(ret_df); ann.index = ann.index.year
    x, bw = np.arange(len(ann)), 0.15
    for k, col in enumerate(ret_df.columns):
        ax7.bar(x+k*bw, ann[col]*100, bw, color=colors[col], label=col, alpha=0.85)
    ax7.set_xticks(x+bw*2)
    ax7.set_xticklabels(ann.index, color="white", fontsize=8)
    ax7.axhline(0, color="white", lw=0.5)
    style(ax7, "Annual Returns (%)")
    ax7.set_ylabel("Return %", color="#aaaacc")
    ax7.legend(framealpha=0, labelcolor="white", fontsize=6)

    fig.suptitle("Riemannian Manifold Portfolio Allocation — 50 Stocks",
                 color="white", fontsize=15, fontweight="bold", y=0.99)
    plt.show()
    print("Plot saved.")


prices = load_data()
ret_df, rb_df, names = run_backtest(prices)

print("\n Performance Summary \n")
print_summary(ret_df)

print("\n Annual Returns \n")
print_annual(ret_df)

print(f"\n Covariance Geometry")
print(f"  Mean cond(Σ_riem):  {rb_df['cond_riem'].mean():.1f}")
print(f"  Mean cond(Σ_eucl):  {rb_df['cond_eucl'].mean():.1f}")
print(f"  Stress periods:     {rb_df['is_stress'].mean():.1%}")

plot_results(ret_df, rb_df, names)