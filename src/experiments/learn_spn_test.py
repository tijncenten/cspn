import numpy as np
from numpy.random.mtrand import RandomState
import matplotlib.pyplot as plt

import math

from node import Sum, Product, NDProduct, Leaf, Bernoulli, Categorical, Gaussian
from utils import get_nodes, get_topological_order_layers, sample, mpe, gradient_backward, sgd, add_ids_to_spn
from learning import generate_dense_spn, random_region_graph, initialize_weights, learn_spn
from mcmc import *

from scipy.stats import *

# SPN training gaussian

def run_test():
    def test_gaussian(spn, data):
        cor_count = 0
        tot_count = 0
        for d in data:
            evidence = d.copy()
            label = evidence[-1]
            evidence[-1] = np.nan

            spn.value(evidence=evidence, ll=True)
            lls = [c._ll + np.log(w) - spn._ll for c, w in zip(spn.children, spn.weights)]
            guess = np.argmax(lls)

            # s = mpe(spn, evidence, rand_gen=RandomState())
            # guess = s[-1]

            if int(guess) == int(label):
                cor_count += 1
            tot_count += 1
        print(cor_count, tot_count, cor_count / tot_count)

    np.random.seed(123)
    data = np.c_[np.r_[np.random.normal(6, 1, (500, 2)), np.random.normal(9, 1, (500, 2))],
                    np.r_[np.zeros((500, 1)), np.ones((500, 1))]]
    rvs = [0,1,2]

    epochs = 1
    lr = 0.8
    gaussian_spn_l = learn_spn(data[:500], rvs, epochs=epochs, lr=lr, leaf_types=[Gaussian, Gaussian, Bernoulli])
    gaussian_spn_r = learn_spn(data[500:], rvs, epochs=epochs, lr=lr, leaf_types=[Gaussian, Gaussian, Bernoulli])
    gaussian_spn = Sum(weights=[0.5, 0.5], children=[gaussian_spn_l, gaussian_spn_r])

    print()
    print('Gaussian SPN:')

    test_gaussian(gaussian_spn, data)


if __name__ == '__main__':
    run_test()
