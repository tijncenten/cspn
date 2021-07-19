from spn.algorithms.Validity import is_valid
from spn.structure.leaves.parametric.Parametric import Categorical, Gaussian
from spn.structure.Base import Context, Sum, Product
from spn.structure.Base import assign_ids, rebuild_scopes_bottom_up

from algorithms.graphics import plot_labeled_spn
from algorithms.learning import learn_class_discriminative

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

x0_mean = 5
x0_std = 1

x1_0_mean = 3
x1_0_std = 0.5

x1_1_mean = 5
x1_1_std = 0.5

def create_structure():
    x0 = Gaussian(mean=x0_mean, stdev=x0_std, scope=0)
    x1_0 = Gaussian(mean=x1_0_mean, stdev=x1_0_std, scope=1)
    x1_1 = Gaussian(mean=x1_1_mean, stdev=x1_1_std, scope=1)
    s = Sum(weights=[0.5, 0.5], children=[x1_0, x1_1])
    spn = Product(children=[x0, s])

    assign_ids(spn)
    rebuild_scopes_bottom_up(spn)

    return spn


spn = create_structure()
assert is_valid(spn)

large_plot = False
margins = 1

fig = plt.figure()

plot_labeled_spn(spn,
    f'src/report-vis/spn-example.png',
    large=large_plot,
    save=False,
    margins=margins
)

fig.text(0.41, 0.4, '0.5')
fig.text(0.48, 0.4, '0.5')

# Axis of X_1 left
ax_x1_0 = fig.add_axes([0.2, 0.085, 0.2, 0.2])

x_x1 = np.arange(1, 7, 0.01)
y_x1_0 = norm.pdf(x_x1, x1_0_mean, x1_0_std)
ax_x1_0.plot(x_x1, y_x1_0)
ax_x1_0.set_xlim(left=1, right=7)
ax_x1_0.set_xlabel('$X_1$', loc='right', labelpad=0)

# Axis of X_1 right
ax_x1_1 = fig.add_axes([0.5, 0.085, 0.2, 0.2])

y_x1_1 = norm.pdf(x_x1, x1_1_mean, x1_1_std)
ax_x1_1.plot(x_x1, y_x1_1)
ax_x1_1.set_xlim(left=1, right=7)
ax_x1_1.set_xlabel('$X_1$', loc='right', labelpad=0)

# Axis of sum node
ax_sum = fig.add_axes([0.15, 0.5, 0.2, 0.2])

y_sum = y_x1_0 * 0.5 + y_x1_1 * 0.5
ax_sum.plot(x_x1, y_sum)
ax_sum.set_xlim(left=1, right=7)
ax_sum.set_xlabel('$X_1$', loc='right', labelpad=0)


# Axis of X_0
ax_x0 = fig.add_axes([0.75, 0.4, 0.2, 0.2])

x_x0 = np.arange(2, 8, 0.01)
y_x0 = norm.pdf(x_x0, x0_mean, x0_std)
ax_x0.plot(x_x0, y_x0)
ax_x0.set_xlim(left=2, right=8)
ax_x0.set_xlabel('$X_0$', loc='right', labelpad=0)

# Axis of product node (root)
ax_product = fig.add_axes([0.46, 0.75, 0.2, 0.2])

def f(x,y):
    s = norm.pdf(x, x1_0_mean, x1_0_std) * 0.5 + norm.pdf(x, x1_1_mean, x1_1_std) * 0.5
    x0 = norm.pdf(y, x0_mean, x0_std)
    return s * x0

X, Y = np.meshgrid(x_x1, x_x0)
print(X.shape)
print(Y.shape)
Z = f(X,Y)
ax_product.contour(X, Y, Z)
ax_product.set_xlim(left=1, right=7)
ax_product.set_ylim(bottom=2, top=8)
ax_product.set_xlabel('$X_1$', loc='right', labelpad=0)
ax_product.set_ylabel('$X_0$', loc='top', labelpad=0)

fig.savefig(f'src/report-vis/spn-example.png')
fig.savefig(f'src/report-vis/spn-example.pdf')

