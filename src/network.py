import numpy as np
from numpy.random.mtrand import RandomState
import matplotlib.pyplot as plt

import math

from node import Sum, Product, NDProduct, Leaf, Bernoulli, Categorical, Gaussian
from utils import get_nodes, get_topological_order_layers, sample, mpe, gradient_backward, sgd, add_ids_to_spn
from learning import generate_dense_spn, random_region_graph, initialize_weights, learn_spn
from mcmc import *
import time

from scipy.stats import *

from experiments.settings import Settings
from experiments import learn_spn_test, spflow_rob_test, plot_rob_results, spflow_cycle_test, spflow_rat_spn_test, spflow_rat_spn_test2, spflow_mnist_test
from experiments import analyze_data, spflow_rat_spn_vis

from spn.experiments.RandomSPNs_layerwise.distributions import RatNormal
from spn.experiments.RandomSPNs_layerwise.rat_spn import RatSpnConfig

config = RatSpnConfig()
config.F = None # Number of input features/variables (filled in later)
config.R = 2 # Number of repetitions
config.D = 2 # The depth
config.I = 1 # Number of distributions for each scope at the leaf layer
config.S = 2 # Number of sum nodes at each layer
config.C = 1 # The number of classes
config.dropout = 0.0
config.leaf_base_class = RatNormal
config.leaf_base_kwargs = {}

config2 = RatSpnConfig()
config2.F = None # Number of input features/variables (filled in later)
config2.R = 1 # Number of repetitions
config2.D = 2 # The depth
config2.I = 1 # Number of distributions for each scope at the leaf layer
config2.S = 2 # Number of sum nodes at each layer
config2.C = 1 # The number of classes
config2.dropout = 0.0
config2.leaf_base_class = RatNormal
config2.leaf_base_kwargs = {}

n_epochs = 60
batch_size = 100
learning_rate = 1e-2
min_instances_slice = 200 # SPFlow default: 200
# leaf_eps = 0.1
# leaf_eps_list = [2.0, 4.0, 6.0] # [None, 0.1, 0.2, 0.5, 1.0]
# leaf_eps_list = [None, 0.1, 0.2, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0]
# leaf_eps_list = [None, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0]

# leaf_eps_list = [None, 0.1, 0.5, 1.0, 2.0, 4.0, 6.0]
leaf_eps_list = [None]
# norm = None
norm = 'zscore'

dataset = 'diabetes' # 'robot'

settings_list = []
settings_list.append(Settings(dataset, min_instances_slice=min_instances_slice, norm=norm, rat_spn_large=False))

settings_list.append(Settings(dataset, min_instances_slice=min_instances_slice, norm=norm, rat_spn_large=False, class_discriminative=True))

# for leaf_eps in leaf_eps_list:
#     settings_list.append(
#         Settings(dataset, class_discriminative=True,
#             build_rat_spn=True, rat_spn_config=config, rat_spn_large=False,
#             n_epochs=n_epochs, batch_size=batch_size, learning_rate=learning_rate, norm=norm, leaf_eps=leaf_eps)
#     )

settings_list.append(
    Settings(dataset, class_discriminative=True,
        build_rat_spn=True, rat_spn_config=config, rat_spn_large=False,
        n_epochs=n_epochs, batch_size=batch_size, learning_rate=learning_rate, norm=norm)
)


# settings_list.append(
#     Settings(dataset, class_discriminative=True,
#         build_rat_spn=True, rat_spn_config=config, rat_spn_large=False,
#         n_epochs=n_epochs, batch_size=batch_size, learning_rate=learning_rate, norm='zscore', leaf_eps=0.1)
# )
# settings_list.append(
#     Settings(dataset, class_discriminative=True,
#         build_rat_spn=True, rat_spn_config=config, rat_spn_large=False,
#         n_epochs=n_epochs, batch_size=batch_size, learning_rate=learning_rate, norm='zscore', leaf_eps=0.2)
# )
# settings_list.append(
#     Settings(dataset, class_discriminative=True,
#         build_rat_spn=True, rat_spn_config=config, rat_spn_large=False,
#         n_epochs=n_epochs, batch_size=batch_size, learning_rate=learning_rate, norm='zscore', leaf_eps=0.5)
# )
# settings_list.append(
#     Settings(dataset, class_discriminative=True,
#         build_rat_spn=True, rat_spn_config=config, rat_spn_large=False,
#         n_epochs=n_epochs, batch_size=batch_size, learning_rate=learning_rate, leaf_eps=leaf_eps)
# )


# settings_list.append(
#     Settings(dataset, class_discriminative=True,
#         build_rat_spn=True, rat_spn_config=config2, rat_spn_large=True,
#         n_epochs=n_epochs, batch_size=batch_size, learning_rate=learning_rate, norm='zscore')
# )
# settings_list.append(
#     Settings(dataset, class_discriminative=True,
#         build_rat_spn=True, rat_spn_config=config2, rat_spn_large=True,
#         n_epochs=n_epochs, batch_size=batch_size, learning_rate=learning_rate, leaf_eps=0.1)
# )


start = time.time()

# spflow_rat_spn_test.run_test()
# spflow_rat_spn_test2.run_test(settings_list[0])
# spflow_rat_spn_vis.run_test(settings_list[0])
# analyze_data.run_test()
# Run for testing cycle detection algorithm and min/max constraints algorithm
# spflow_cycle_test.run_test()


# Robustness evaluation run
for settings in settings_list:
    spn = None
    for leaf_eps in leaf_eps_list:
        settings.leaf_eps = leaf_eps
        spn = spflow_rob_test.run_test(settings, spn=spn)

end = time.time()
print(f'time: {end - start}')
