import numpy as np
from numpy.random.mtrand import RandomState
import matplotlib.pyplot as plt

import math

from node import Sum, Product, NDProduct, Leaf, Bernoulli, Categorical, Gaussian
from utils import get_nodes, get_topological_order_layers, sample, mpe, gradient_backward, sgd, add_ids_to_spn
from learning import generate_dense_spn, random_region_graph, initialize_weights, learn_spn
from mcmc import *

from scipy.stats import *

print('\n\nTraining with SPFlow\n')

# train_data = np.array(data)

np.random.seed(123)
train_data = np.c_[np.r_[np.random.normal(6, 1, (500, 2)), np.random.normal(9, 1, (500, 2))],
                   np.r_[np.zeros((500, 1)), np.ones((500, 1))]]

from spn.algorithms.LearningWrappers import learn_parametric, learn_classifier
from spn.structure.leaves.parametric.Parametric import Categorical, Gaussian
from spn.structure.Base import Context
spn_classification = learn_classifier(train_data,
    Context(parametric_types=[Gaussian, Gaussian, Categorical]).add_domains(train_data),
    learn_parametric, 2)

# test_data = data.copy()
# for d in test_data:
#         d[-1] = np.nan

# test_data = np.array(test_data)


test_data = np.c_[np.r_[np.random.normal(6, 1, (500, 2)), np.random.normal(9, 1, (500, 2))],
                   np.r_[np.zeros((500, 1)), np.ones((500, 1))]]

test_data[:,2] = np.nan

# test_data = np.array([3.0, 4.0, np.nan, 12.0, 18.0, np.nan]).reshape(-1, 3)

print(train_data)
print(test_data)

from spn.algorithms.MPE import mpe
res = mpe(spn_classification, test_data)
res = res[:,2]

print(np.sum(res[:500]))
print(np.sum(res[500:]))

from spn.io.Graphics import plot_spn

plot_spn(spn_classification, './spn.png')
