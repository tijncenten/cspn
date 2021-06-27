import numpy as np
from numpy.random.mtrand import RandomState
import matplotlib.pyplot as plt

import math

from node import Sum, Product, NDProduct, Leaf, Bernoulli, Categorical, Gaussian
from utils import get_nodes, get_topological_order_layers, sample, gradient_backward, sgd, add_ids_to_spn
from learning import generate_dense_spn, random_region_graph, initialize_weights, learn_spn
from mcmc import *

from scipy.stats import *

p0 = Product(children=[Bernoulli(p=0.7, scope=1), Bernoulli(p=0.6, scope=2)])
p1 = Product(children=[Bernoulli(p=0.5, scope=1), Bernoulli(p=0.4, scope=2)])
s1 = Sum(weights=[0.3, 0.7], children=[p0, p1])
p2 = Product(children=[Bernoulli(p=0.8, scope=0), s1])
p3 = Product(children=[Bernoulli(p=0.8, scope=0), Bernoulli(p=0.7, scope=1)])
p4 = Product(children=[p3, Bernoulli(p=0.6, scope=2)])
spn = Sum(weights=[0.4, 0.6], children=[p2, p4])

print(spn)



print(np.exp(spn.value(evidence=[np.nan, np.nan, np.nan], ll=True)))
print(np.exp(spn.value(evidence=[0, np.nan, np.nan], ll=True)))
print(np.exp(spn.value(evidence=[np.nan, np.nan, 0], ll=True)))
print(np.exp(spn.value(evidence=[0, np.nan, 0], ll=True)))
