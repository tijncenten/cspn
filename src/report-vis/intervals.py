
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# interval dominance

fig, ax = plt.subplots(figsize=(3,4))

intervals = [(0.4, 0.5), (0.25, 0.45), (0.05, 0.35)]
colors = ['blue', 'green', 'red']

classes = ['c1', 'c2', 'c3']
values_low = [low for low, high in intervals]
values_size = [high - low for low, high in intervals]
ax.bar(classes, values_size, bottom=values_low, width=0.5, color=colors)
ax.set_ylim(bottom=-0.05, top=0.65)
fig.savefig('src/report-vis/interval-dominance.png')
fig.savefig('src/report-vis/interval-dominance.pdf')

plt.close()

# credal dominance

fig, ax = plt.subplots(figsize=(3,4))

intervals = [(0.4, 0.5), (0.25, 0.45), (0.05, 0.35)]

c1 = [0.4, 0.4, 0.5, 0.5]
c2 = [0.35, 0.25, 0.35, 0.45]
c3 = [0.25, 0.35, 0.15, 0.05]

cs = [c1, c2, c3]
e1 = [ci[0] for ci in cs]
e2 = [ci[1] for ci in cs]
e3 = [ci[2] for ci in cs]
e4 = [ci[3] for ci in cs]

classes = ['e1', 'e2', 'e3', 'e4']
values = [0 for c in classes]
ax.bar(classes, values, width=0.5)

x_pos = ax.get_xticks()
[plt.scatter(x, y, s=10, color=colors[0]) for y, x in zip(c1, x_pos)]
ax.plot(x_pos, c1, color=colors[0], label='c1')
[plt.scatter(x, y, s=10, color=colors[1]) for y, x in zip(c2, x_pos)]
ax.plot(x_pos, c2, color=colors[1], label='c2')
[plt.scatter(x, y, s=10, color=colors[2]) for y, x in zip(c3, x_pos)]
ax.plot(x_pos, c3, color=colors[2], label='c3')

ax.legend()

ax.set_ylim(bottom=-0.05, top=0.65)


fig.savefig('src/report-vis/credal-dominance.png')
fig.savefig('src/report-vis/credal-dominance.pdf')


plt.close()
