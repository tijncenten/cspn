import numpy as np
from numpy.random.mtrand import RandomState
import matplotlib.pyplot as plt

import math

from node import Sum, Product, NDProduct, Leaf, Bernoulli, Categorical, Gaussian
from utils import get_nodes, get_topological_order_layers, sample, gradient_backward, sgd, add_ids_to_spn
from learning import generate_dense_spn, random_region_graph, initialize_weights, learn_spn
from mcmc import *

from scipy.stats import *

# TODO: Cleanup file by dividing tests

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

exit()



print(spn)

print(np.exp(spn.value(evidence=[np.nan, np.nan], ll=True)))
print(np.exp(spn.value(evidence=[5, np.nan], ll=True)))
print(np.exp(spn.value(evidence=[np.nan, 10], ll=True)))
print(np.exp(spn.value(evidence=[5, 10], ll=True)))
print(np.exp(spn.value(evidence=[10, 5], ll=True)))



# print(np.exp(spn.value(evidence=[np.nan, np.nan, np.nan], ll=True)))
# print(np.exp(spn.value(evidence=[0, np.nan, np.nan], ll=True)))
# print(np.exp(spn.value(evidence=[np.nan, np.nan, 0], ll=True)))
# print(np.exp(spn.value(evidence=[0, np.nan, 0], ll=True)))
# print('non-decomposable:')
# print(np.exp(spn.value(evidence=[np.nan, 0, np.nan], ll=True)))

exit()
##### Test non-decomposable product node #####




p0 = Product(children=[Bernoulli(p=0.7, scope=1), Bernoulli(p=0.6, scope=2)])
p1 = Product(children=[Bernoulli(p=0.5, scope=1), Bernoulli(p=0.4, scope=2)])
s1 = Sum(weights=[0.3, 0.7], children=[p0, p1])
p2 = Product(children=[Bernoulli(p=0.8, scope=0), s1])
p3 = Product(children=[Bernoulli(p=0.8, scope=0), Bernoulli(p=0.7, scope=1)])
p4 = Product(children=[p3, Bernoulli(p=0.6, scope=2)])
spn = Sum(weights=[0.4, 0.6], children=[p2, p4])

print(spn)
print()

x = [sample(spn, np.array([np.nan, np.nan, np.nan]), RandomState(i)) for i in range(100)]
print(sum(x) / len(x))

train_data = [[1, 1, 1]]

epochs = 1000
for e in range(epochs):
    for d in train_data:
        ll = spn.value(d, ll=True)
        gradient_backward(spn)
        sgd(spn, lr=0.01, data=d)
    if e % 100 == 0:
        print(e, ll)


x = [sample(spn, np.array([np.nan, np.nan, np.nan]), RandomState(i)) for i in range(100)]
print(sum(x) / len(x))

print()
print(spn)

exit()





print(spn.value(evidence=[np.nan, np.nan, np.nan]))
print(spn.value(evidence=[0, np.nan, np.nan]))
print(spn.value(evidence=[1, np.nan, np.nan]))
print(spn.value(evidence=[np.nan, 0, np.nan]))
print(spn.value(evidence=[np.nan, 1, np.nan]))
print(spn.value(evidence=[np.nan, np.nan, 0]))
print(spn.value(evidence=[np.nan, np.nan, 1]))
print()

# The likelihood of all possible options of evidence should be (roughly) equal to 1
x = 0
x += spn.value(evidence=[0, 0, 0])
x += spn.value(evidence=[0, 0, 1])
x += spn.value(evidence=[0, 1, 0])
x += spn.value(evidence=[0, 1, 1])
x += spn.value(evidence=[1, 0, 0])
x += spn.value(evidence=[1, 0, 1])
x += spn.value(evidence=[1, 1, 0])
x += spn.value(evidence=[1, 1, 1])
print(x)
print()

# Compare the log likelihood to that of SPFlow's documentation
print('Computing log likelihood')
ll = spn.value(evidence=[1, 0, 1], ll=True)
print(ll, np.exp(ll)) # Should be [[-1.90730501]] [[0.14848]]
print()

# Compare the log likelihood of a marginalization to that of SPFlow's documentation
print('Marginal inference')
ll_mar = spn.value(evidence=[np.nan, 0, 1], ll=True)
print(ll_mar, np.exp(ll_mar)) # Should be [[-1.68416146]] [[0.1856]]
print()


print(spn.value(evidence=[np.nan, np.nan, 1]))


print(spn == spn)
print(spn == p4)
print(spn == s1)

print()
print(get_nodes(spn))

print(get_topological_order_layers(spn))

# def gen_sample(i):
#     s = sample(spn, np.array([np.nan, 0, 0]), RandomState(i))
#     print('sample', s)

# for i in range(10):
#     gen_sample(i)

# print()

# s = sample(spn, np.array([np.nan, np.nan, np.nan]), RandomState(123))
# print('sample', s)

# s = sample(spn, np.array([np.nan, np.nan, np.nan]), RandomState(123))
# print('sample', s)

print('\n\n')


spn2 = Sum(weights=[0.5, 0.5], children=[Bernoulli(1, scope=0), Bernoulli(0, scope=0)])
# spn2 = Bernoulli(0.5, scope=0)

print(spn2)
print()

ll_mar = spn2.value(evidence=[0], ll=True)
print('likelihood of 0: ', ll_mar, np.exp(ll_mar))
ll_mar = spn2.value(evidence=[1], ll=True)
print('likelihood of 1: ', ll_mar, np.exp(ll_mar))


print()
x = [sample(spn2, np.array([np.nan]), RandomState(i))[0] for i in range(100)]
print(sum(x) / len(x))
print()

# Generate 100 data samples
np.random.seed(1)
data = np.random.binomial(1, p=0.2, size=100)
test = np.random.binomial(1, p=0.2, size=20)
np.random.seed(None)

print('data p: ', sum(data) / len(data))

print(spn2)

print('\n')

iterations = 0 # 100
# Perform iterations
for i in range(iterations):

    # Train dataset
    for di, d in enumerate(data):
        ll = spn2.value(evidence=np.array([d]), ll=True)
        # if di % 25 == 0:
        #     print('\t', di, ll, np.exp(ll))
        gradient_backward(spn2)
        sgd(spn2, lr=0.01 / (i+1))
    
    # Test dataset
    test_ll = 0
    for t in test:
        ll = spn2.value(evidence=np.array([t]), ll=True)
        test_ll += ll
    print(f'Epoch {i}', 'test ll: ', test_ll / len(test))
    x = [sample(spn2, np.array([np.nan]), RandomState(i))[0] for i in range(100)]
    print(sum(x) / len(x))

print()
x = [sample(spn2, np.array([np.nan]), RandomState(i))[0] for i in range(100)]
print(sum(x) / len(x))
print()

print(spn2)


# exit()


# Test code for generating dense spn
rvs = [0,1,2,3,4,5,6]
# rvs = [0,1]
region_graph = random_region_graph(rvs, depth=2, repetitions=2)

print(region_graph)

constructed_spn = region_graph.to_spn()

add_ids_to_spn(constructed_spn)

initialize_weights(constructed_spn)

print(constructed_spn)


#####################
# SPN Learning test #
#####################

def test_xor(spn, data):
    for d in data:
        evidence = d.copy()
        evidence[-1] = np.nan
        s = sample(spn, evidence, rand_gen=RandomState())
        print(s)

# Train the XOR table
rvs = [0,1,2]
data = [
    [0,0,0],
    [0,1,1],
    [1,0,1],
    [1,1,0]
]

print()
print('Learning SPN on XOR table')
xor_spn = learn_spn(data, rvs, epochs=100, lr=0.0001)
print()
print('XOR SPN:')
# print(xor_spn)


test_xor(xor_spn, data)

print(data)
