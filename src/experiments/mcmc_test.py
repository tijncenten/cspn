import numpy as np
from numpy.random.mtrand import RandomState
import matplotlib.pyplot as plt

import math

from node import Sum, Product, NDProduct, Leaf, Bernoulli, Categorical, Gaussian
from utils import get_nodes, get_topological_order_layers, sample, gradient_backward, sgd, add_ids_to_spn
from learning import generate_dense_spn, random_region_graph, initialize_weights, learn_spn
from mcmc import *

from scipy.stats import *

# Monte Carlo integration example 1D
# def f(x, mean=0):
#     return norm.pdf(x, loc=mean, scale=1)

# N = 10000

# x = uniform.rvs(loc=0, scale=10, size=N)
# print(np.mean(x))

# F = np.mean([f(xi, mean=5) / uniform.pdf(xi, loc=0, scale=10) for xi in x])
# print(F)

# # Monte Carlo integration example 2D
# def f(x, y, mean=0, mean2=7):
#     return norm.pdf(x, loc=mean, scale=1) * norm.pdf(y, loc=mean2, scale=1)

# N = 10000

# x = uniform.rvs(loc=0, scale=10, size=(N,2))
# print(np.mean(x, axis=0))

# F = np.mean([f(xi[0], xi[1], mean=5) / (uniform.pdf(xi[0], loc=0, scale=10) * uniform.pdf(xi[1], loc=0, scale=10)) for xi in x])
# print(F)





# l1 = Leaf(scope=0)
# l2 = Leaf(scope=1)

# l3 = Leaf(scope=0)
# l4 = Leaf(scope=1)

# p1 = Product(children=[l1, l2])
# p2 = Product(children=[l3, l4])

# s1 = Sum(children=[p1, p2], weights=[0.3, 0.7])

# print(s1)


##### Test MCMC #####

# n = 50000
# burn_in = 0
# random_samples = mcmc_generator(n)
# print(f'n: {n}')

# print(np.sum([r*r for r in random_samples[burn_in:]]) / (n - burn_in))
# print(np.mean(random_samples))
# print(np.median(random_samples))

# plt.title('histogram')
# plt.hist(random_samples, bins=30)
# plt.grid(True)
# plt.savefig('test.png')
# plt.close()

# print(np.mean([r*r for r in random_samples]))

# plt.title('histogram')
# plt.hist([r*r for r in random_samples], bins=30, range=(0, 10))
# plt.grid(True)
# plt.savefig('test2.png')

# exit()
##### Test MCMC #####

##### Test non-decomposable product node #####

# b1 = Bernoulli(p=0.5, scope=0)
# b2 = Bernoulli(p=0.5, scope=1)
# b4 = Bernoulli(p=0.5, scope=2)
# p0 = Product(children=[b1, b2])
# p1 = Product(children=[b2, b4])
# p3 = Product(children=[p0, p1])

# spn = p3


# p0 = Product(children=[Bernoulli(p=0.7, scope=1), Bernoulli(p=0.6, scope=2)])
# p1 = Product(children=[Bernoulli(p=0.5, scope=1), Bernoulli(p=0.4, scope=2)])
# s1 = Sum(weights=[0.3, 0.7], children=[p0, p1])
# p2 = Product(children=[Bernoulli(p=0.8, scope=0), s1])
# p3 = Product(children=[Bernoulli(p=0.8, scope=0), Bernoulli(p=0.7, scope=1)])
# p4 = Product(children=[p3, Bernoulli(p=0.6, scope=2)])
# spn = Sum(weights=[0.4, 0.6], children=[p2, p4])


g0 = Gaussian(mean=5, stdev=1, scope=0)
g1 = Gaussian(mean=9, stdev=2, scope=0)
s0 = Sum(weights=[0.3, 0.7], children=[g0, g1])
g2 = Gaussian(mean=7.5, stdev=2, scope=0)

# p0 = NDProduct(children=[s0, g2])
p0 = Product(children=[s0, g2])

spn = p0

samples = [sample(spn, [np.nan], RandomState())[0] for i in range(10000)]


def sc0(x):
    return norm.pdf(x, loc=5, scale=1) * 0.3 + norm.pdf(x, loc=9, scale=2) * 0.7

def sc(x):
    return sc0(x) * norm.pdf(x, loc=7.5, scale=2) / 0.1144

plt.title('SPN Samples (N=10000)')
plt.hist(samples, bins=30, histtype='bar', ec='black', density=True)
x = np.linspace(0, 15, 1000)
plt.plot(x, sc(x))
plt.grid(True)
plt.savefig('spn-samples.png')
plt.close()
