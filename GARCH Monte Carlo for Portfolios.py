import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from arch import arch_model
from scipy.optimize import minimize
from numpy.random import Generator, PCG64

tickers = ["AAPL", "MSFT", "GOOG"]
weights = np.array([0.4, 0.3, 0.3])
df = yf.download(tickers, period='5y', auto_adjust=True)
returns = np.log(df['Close'] / df['Close'].shift(1)).dropna()

# Fit univariate GARCH(1,1)
garch_res = {}
std_resids = pd.DataFrame(index=returns.index, columns=returns.columns, dtype=float)
cond_vol = pd.DataFrame(index=returns.index, columns=returns.columns, dtype=float)
params_univ = {}

for col in returns.columns:
    am = arch_model(returns[col]*100, vol='Garch', p=1, q=1, mean='Constant', dist='normal')

    # scale returns so numeric is stable (percent)
    r = am.fit(disp='off')
    garch_res[col] = r
    std_resids[col] = r.std_resid.values
    cond_vol[col] = r.conditional_volatility.values / 100.0  # back to original scale
    p = r.params

    # Common naming in arch: 'omega', 'alpha[1]', 'beta[1]'
    def get_param_like(idx_list, series):
        for k in idx_list:
            if k in series.index:
                return float(series[k])
        raise KeyError("param not found")

    omega = get_param_like(['omega'], p)
    alpha = get_param_like(['alpha[1]', 'alpha[0]'], p)
    beta  = get_param_like(['beta[1]', 'beta[0]'], p)
    params_univ[col] = {'omega': omega/10000.0,  # because we scaled returns by 100
                        'alpha': alpha,
                        'beta': beta}

# standardized residuals matrix Z (T x N)
Z = std_resids.values.astype(float)
T2, N2 = Z.shape

# Estimate DCC(1,1) parameters (a,b) by maximizing DCC log-likelihood
Qbar = np.cov(Z.T)

def dcc_loglik(theta):
    a, b = theta

    if a < 0 or b < 0 or a + b >= 1:
        return 1e9

    Qt = Qbar.copy()
    ll = 0.0

    for t in range(T2):
        if t == 0:
            zprev = Z[t]  # use t as prev when t=0
        else:
            zprev = Z[t-1]

        zcol = zprev[:, None]
        Qt = (1 - a - b) * Qbar + a * (zcol @ zcol.T) + b * Qt

        # normalize to correlation
        diag = np.sqrt(np.diag(Qt))
        denom = np.outer(diag, diag)
        Rt = Qt / denom

        # numerical stabilize
        Rt = (Rt + Rt.T) / 2

        # compute loglik for observation Z[t]
        try:
            sign, logdet = np.linalg.slogdet(Rt)
            invR = np.linalg.inv(Rt)
        except np.linalg.LinAlgError:
            return 1e9

        zt = Z[t]
        ll += 0.5 * (np.log(2 * np.pi) * N2 + logdet + zt.T @ invR @ zt)
    return ll

# initial guess
init = np.array([0.02, 0.97])
res = minimize(dcc_loglik, init, bounds=[(1e-6,0.999),(1e-6,0.999)], method='L-BFGS-B')

if not res.success:
    raise RuntimeError("DCC estimation failed: " + res.message)
a_hat, b_hat = res.x

# compute full series of Rt and Qt (store last as initial for simulation)
Qt = Qbar.copy()
Q_ts = []
R_ts = []

for t in range(T2):
    if t == 0:
        zprev = Z[t]
    else:
        zprev = Z[t-1]

    zcol = zprev[:, None]
    Qt = (1 - a_hat - b_hat) * Qbar + a_hat * (zcol @ zcol.T) + b_hat * Qt
    diag = np.sqrt(np.diag(Qt))
    Rt = Qt / np.outer(diag, diag)
    Qt = (Qt + Qt.T) / 2
    Rt = (Rt + Rt.T) / 2
    Q_ts.append(Qt.copy())
    R_ts.append(Rt.copy())

# use last observed vol and last R as starting point
last_sigma = cond_vol.iloc[-1].values  # N vector
last_Q = Q_ts[-1]
last_R = R_ts[-1]

# Monte Carlo simulation using DCC + individual GARCH recursions
def simulate_paths(horizon=20, sims=10000, seed=123):
    rng = np.random.default_rng(seed)
    sims = int(sims)
    horizon = int(horizon)

    # arrays: sims x horizon x N
    returns_sim = np.zeros((sims, horizon, N2))

    # initialize
    sigma = np.tile(last_sigma[None, :], (sims, 1))  # sims x N
    Qt_sim = np.tile(last_Q[None, :, :], (sims, 1, 1))
    Rt_sim = np.tile(last_R[None, :, :], (sims, 1, 1))

    # precompute vectorised GARCH params
    omega_vec = np.array([params_univ[col]['omega'] for col in returns.columns])
    alpha_vec = np.array([params_univ[col]['alpha'] for col in returns.columns])
    beta_vec  = np.array([params_univ[col]['beta']  for col in returns.columns])

    for t in range(horizon):
        # draw correlated z for each simulation: sample from MVN(0, Rt) separately per sim
        # we can sample by multiplying standard normals by chol(Rt)
        # vectorised batched Cholesky across all sims simultaneously
        chols = np.linalg.cholesky(Rt_sim)  # sims x N x N
        raw = rng.standard_normal((sims, N2))
        z_t = np.einsum('sij,sj->si', chols, raw)

        # eps = sigma * z
        eps_t = sigma * z_t  # element-wise

        # store returns (here mean=0 assumption for simplicity)
        returns_sim[:, t, :] = eps_t

        # update individual sigma via GARCH(1,1) recursion per simulation and asset
        # vectorised across sims and assets simultaneously
        sigma = np.sqrt(np.maximum(omega_vec + alpha_vec * (eps_t**2) + beta_vec * (sigma**2), 1e-12))

        # update DCC Qt and Rt per sim using z_t (standardized residuals)
        # vectorised: outer product z_t[i] z_t[i].T = einsum
        zprev = z_t[:, :, None]  # sims x N x 1
        outer = zprev @ zprev.transpose(0, 2, 1)  # sims x N x N
        Qt_sim = (1 - a_hat - b_hat) * Qbar + a_hat * outer + b_hat * Qt_sim
        diag_vals = np.sqrt(np.diagonal(Qt_sim, axis1=1, axis2=2))  # sims x N
        denom = diag_vals[:, :, None] * diag_vals[:, None, :]  # sims x N x N
        Rt_sim = Qt_sim / denom

        # numerical symmetrize
        Rt_sim = (Rt_sim + Rt_sim.transpose(0, 2, 1)) / 2
    return returns_sim

horizon = 20
returns_sim = simulate_paths(horizon=horizon, sims=10000, seed=123)

# weight asset returns into portfolio returns: sims x horizon
port_returns = returns_sim @ weights

prices_sim = 100.0 * np.exp(np.cumsum(port_returns, axis=1))
confidence_level = 0.95
terminal_returns = port_returns[:, -1]
VaR = -np.percentile(terminal_returns, 100 * (1 - confidence_level))
tail = terminal_returns[terminal_returns <= -VaR]
cvar = -tail.mean()
portfolio_returns = returns.values @ weights

plt.figure(figsize=(10,4))
plt.hist(portfolio_returns, bins=50, alpha=0.75, color='blue')
plt.title(f"Distribution of GARCH Simulated Portfolio Returns ({horizon} days)")
plt.xlabel('Portfolio Return')
plt.ylabel("Frequency")
plt.show()

print(f"Monte Carlo 95% VaR: {VaR:.2%}")
print(f"Monte Carlo Expected Shortfall: {cvar:.2%}")