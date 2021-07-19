
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Discrete soft evidence

fig, ax = plt.subplots()

factor = 0.2

classes = [0, 1, 2, 3, 4]
values = np.array([0, 0, 1, 0, 0])
values = values * (1 - factor)
se_values = factor / len(classes)
# values += factor / len(classes)
ax.bar(classes, values)
ax.bar(classes, se_values, bottom=values)
ax.set_ylim(bottom=-0.05, top=1.05)

fig.savefig('src/report-vis/soft-evi-discrete.png')
fig.savefig('src/report-vis/soft-evi-discrete.pdf')

plt.close()


# Continuous soft evidence

mean = 0.5
stdev = 0.1

x = np.arange(0.0, 1.0, 0.01)
y = norm.pdf(x, mean, stdev)

obs = 0.4
factor = 1
obs_min = obs - stdev * factor
obs_max = obs + stdev * factor

x_obs = np.arange(obs_min, obs_max, 0.01)
y_obs = norm.pdf(x_obs, mean, stdev)

fig, ax = plt.subplots()
ax.plot(x, y, color='black')

ax.plot(x_obs, y_obs, color='blue')
ax.fill_between(x_obs, y_obs, color='blue', alpha=0.5)
ax.plot([obs, obs], [0.0, norm.pdf(obs, mean, stdev)], color='blue')

prob = norm.cdf(obs_max, mean, stdev) - norm.cdf(obs_min, mean, stdev)
print(f'continuous prob: {prob}')

fig.savefig('src/report-vis/soft-evi.png')
fig.savefig('src/report-vis/soft-evi.pdf')
