import numpy as np
import matplotlib.pyplot as plt

FCF0 = 100
T = 5
growth_forecast = 0.05
sigma = 3

WACC, w_sd = 0.09, 0.01
FCF0, f_sd = 100, 10

w_vals = np.linspace(WACC - sigma*w_sd, WACC + sigma*w_sd, 33*sigma)
f_vals = np.linspace(FCF0 - sigma*f_sd, FCF0 + sigma*f_sd, 33*sigma)

W, F = np.meshgrid(w_vals, f_vals)
V = np.zeros_like(W)

for i in range(W.shape[0]):
    for j in range(W.shape[1]):
        present_value = 0
        for k in range(1, T + 1):
            present_value += (F[i, j] / ((1 + W[i, j]) ** (k + 1)))
        V[i, j] = present_value


plt.figure(figsize=(9, 7))
contours = plt.contourf(W, F, V, levels=50, cmap='plasma')
plt.colorbar(contours, label='DCF Value')

print("This model assumes that yearly free cash flow is constant")

plt.xlabel('Discount Rate (WACC)')
plt.ylabel('Free Cash Flow')
plt.title(f'DCF Parameter Space with Normally Distributed Sensitivities {sigma}-sigma')
plt.show()