import yfinance as yf
import numpy as np
import pandas as pd
from scipy.stats import chi2


df = yf.download(tickers="DVN", period="1y", auto_adjust=True)
df = df[['Close']].dropna().reset_index(drop=True)
df['breach_signal'] = 0
df['simple_returns'] = df['Close'].pct_change().dropna()
std = df['simple_returns'].std(ddof=1)
z_score = 1.6448536270
parametric_VaR = -1 * z_score * std
df.loc[df['simple_returns'] < parametric_VaR, 'breach_signal'] = 1

def kupiec_test(exceptions, alpha):
    exceptions = np.asarray(exceptions)
    T = len(exceptions)
    N = exceptions.sum()

    # empirical breach rate
    p_hat = N / T

    # likelihoods
    L0 = (1 - alpha)**(T - N) * (alpha**N)
    L1 = (1 - p_hat)**(T - N) * (p_hat**N)

    # LR statistic
    LR_uc = -2 * np.log(L0 / L1)

    # p-value from chi-square with 1 df
    p_value = 1 - chi2.cdf(LR_uc, df=1)

    return LR_uc, p_value, p_hat

def christoffersen_independence_test(exceptions):
    """
    Christoffersen (1998) independence test for VaR exceptions.
    Input:
      exceptions : array-like of 0/1 (I_t), length T
    Returns dict with:
      n00, n01, n10, n11 : transition counts
      pi01, pi11, pi_hat    : estimated transition probs and overall exception rate
      LR_ind : likelihood-ratio statistic for independence
      p_value: p-value (chi-square, 1 df)
    Notes:
      - Assumes exceptions are aligned (I_t corresponds to same-period realized return vs VaR).
      - Uses log-likelihood with clipping to avoid numerical issues.
    """
    I = np.asarray(exceptions).astype(int)
    if I.ndim != 1 or I.size < 2:
        raise ValueError("exceptions must be a 1D array-like with at least 2 observations")

    # Build transitions between consecutive days: (I[t-1], I[t]) for t=1..T-1
    prev = I[:-1]
    curr = I[1:]

    n00 = int(np.sum((prev == 0) & (curr == 0)))
    n01 = int(np.sum((prev == 0) & (curr == 1)))
    n10 = int(np.sum((prev == 1) & (curr == 0)))
    n11 = int(np.sum((prev == 1) & (curr == 1)))

    # Transition denominators
    denom0 = n00 + n01  # times previous was 0
    denom1 = n10 + n11  # times previous was 1
    total_trans = n00 + n01 + n10 + n11  # should be T-1

    # Estimate transition probabilities (handle zero denom)
    pi01 = n01 / denom0 if denom0 > 0 else np.nan
    pi11 = n11 / denom1 if denom1 > 0 else np.nan

    # Overall exception probability (using full sample)
    T = len(I)
    N = int(I.sum())
    pi_hat = N / T

    # Safe log-likelihood helper (Bernoulli counts). Clip p to avoid log(0)
    def loglike(count_success, count_total, p):
        if count_total == 0:
            return 0.0
        p = np.clip(p, 1e-12, 1 - 1e-12)
        k = count_success
        return k * np.log(p) + (count_total - k) * np.log(1 - p)

    # Log-likelihood under independence (homogeneous Bernoulli with prob = pi_hat)
    logL_ind = loglike(n01 + n11, total_trans, pi_hat)

    # Log-likelihood under first-order Markov (separate transition probs)
    # For transitions from 0: n00+n01 trials, successes = n01 with prob pi01
    logL_markov = 0.0
    logL_markov += loglike(n01, denom0, pi01 if not np.isnan(pi01) else 0.0)
    logL_markov += loglike(n11, denom1, pi11 if not np.isnan(pi11) else 0.0)

    # LR statistic and p-value (df=1)
    LR_ind = -2.0 * (logL_ind - logL_markov)
    # numerical noise may yield tiny negative LR; force non-negative
    LR_ind = max(0.0, LR_ind)
    p_value = 1.0 - chi2.cdf(LR_ind, df=1)

    return {
        "n00": n00, "n01": n01, "n10": n10, "n11": n11,
        "denom0": denom0, "denom1": denom1, "total_trans": total_trans,
        "pi01": pi01, "pi11": pi11, "pi_hat": pi_hat,
        "LR_ind": LR_ind, "p_value": p_value
    }

def christoffersen_conditional_coverage(LR_uc, LR_ind):
    """
    Christoffersen (1998) Conditional Coverage Test:
    LR_cc = LR_uc + LR_ind
    Under H0 ~ Chi-square(df = 2)
    """
    LR_cc = LR_uc + LR_ind
    p_value = 1 - chi2.cdf(LR_cc, df=2)
    return LR_cc, p_value

LR_uc, p_value, p_hat = kupiec_test(df['breach_signal'], 0.05)
results = christoffersen_independence_test(df['breach_signal'])
LR_ind = results['LR_ind']
LR_cc, p_value = christoffersen_conditional_coverage(LR_uc, LR_ind)
