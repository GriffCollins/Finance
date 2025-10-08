import numpy as np
import matplotlib.pyplot as plt

num_scenarios = 10000
forecast_period = 5
required_rate_of_return = 0.1
initial_FCFE = 100
alpha = 0.05

growth_FCFE = 0.1
std_growth_FCFE = 0.05
future_fcfe = initial_FCFE*(1+np.random.normal(growth_FCFE, std_growth_FCFE, size=(num_scenarios, forecast_period)))

present_values = []
for scenario in future_fcfe:
    present_value = 0
    for i, fcfe in enumerate(scenario):
        present_value += fcfe / (1 + required_rate_of_return)**(i+1)
    present_values.append(present_value)

company_value = np.mean(present_values)
lower_bound, upper_bound = np.percentile(present_values, [100*alpha/2, 100* (1-alpha/2)])

print(f'The estimated value of the company {company_value}')

plt.figure(figsize=(12, 8))
plt.hist(present_values, bins=50, alpha=0.5, color='blue')
plt.title(f"Distribution of Simulated DCF with forecast period {forecast_period} years")
plt.axvline(company_value, color='red', linestyle='--', label="Mean")
plt.axvline(upper_bound, color='orange', linestyle='--', label="95% confidence interval")
plt.axvline(lower_bound, color='green', linestyle='--', label="5% confidence interval")
plt.legend()
plt.xlabel("Valuation")
plt.ylabel("Frequency")
plt.show()