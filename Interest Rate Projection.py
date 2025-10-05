from fredapi import Fred
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

fred = Fred(api_key="f022be36d1b0a0ad0437176d04939ad8")
t_bill_3m = fred.get_series("TB3MS")
rates = t_bill_3m.dropna().resample("M").last() / 100

dt_est = 1/12
r = rates.values[:-1]
r_next = rates.values[1:]
delta_r = r_next - r

X = r.reshape(-1, 1)
y = delta_r

reg = LinearRegression().fit(X, y)

beta = reg.coef_[0]
alpha = reg.intercept_

a = -beta / dt_est
theta = alpha / beta
residuals = y - reg.predict(X)
sigma = np.std(residuals) / np.sqrt(dt_est)

T = 5.0
dt_sim = 0.004
N = int(T / dt_sim)
n_paths = 7
r0 = rates[-1]

sigma_sim = sigma * np.sqrt(dt_est / dt_sim)

np.random.seed(42)
paths = np.zeros((n_paths, N))
paths[:, 0] = r0
time = np.linspace(0, T, N)
dt_sqrt = np.sqrt(dt_sim)

for i in range(1, N):
    dz = np.random.normal(0, 1, n_paths)
    dr = a * (theta - paths[:, i-1]) * dt_sim + sigma_sim * dz * dt_sqrt
    paths[:, i] = paths[:, i-1] + dr

plt.figure(figsize=(10, 5))
for i in range(n_paths):
    plt.plot(time, paths[i], label=f'Path {i+1}')
plt.xlabel('Time (Years)')
plt.ylabel('Short Rate')
plt.title('Hull-White Model: Simulated Interest Rate Paths')
plt.grid(True)
plt.legend()
average_path = np.mean(paths, axis=0)
plt.plot(time, average_path, color='black', linewidth=2.5, linestyle='--', label='Average Path')

plt.show()

