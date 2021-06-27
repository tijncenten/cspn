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
from experiments import analyze_data

from spn.experiments.RandomSPNs_layerwise.distributions import RatNormal
from spn.experiments.RandomSPNs_layerwise.rat_spn import RatSpnConfig

config = RatSpnConfig()
config.F = None # Number of input features/variables (filled in later)
config.R = 1 # Number of repetitions
config.D = 2 # The depth
config.I = 2 # Number of distributions for each scope at the leaf layer
config.S = 3 # Number of sum nodes at each layer
config.C = 1 # The number of classes
config.dropout = 0.0
config.leaf_base_class = RatNormal
config.leaf_base_kwargs = {}

config2 = RatSpnConfig()
config2.F = None # Number of input features/variables (filled in later)
config2.R = 3 # Number of repetitions
config2.D = 2 # The depth
config2.I = 4 # Number of distributions for each scope at the leaf layer
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
leaf_eps_list = [None, 0.1, 0.5, 1.0, 2.0, 4.0, 6.0]
# norm = None
norm = 'zscore'

dataset = 'gesture' # 'robot'

settings_list = []
# settings_list.append(Settings(dataset, min_instances_slice=min_instances_slice, norm=norm))
# settings_list.append(Settings(dataset, min_instances_slice=min_instances_slice, leaf_eps=0.1))
# settings_list.append(Settings(dataset, min_instances_slice=min_instances_slice, leaf_eps=0.2))
# for leaf_eps in leaf_eps_list:
#     settings_list.append(Settings(dataset, min_instances_slice=min_instances_slice, norm=norm, leaf_eps=leaf_eps))

# settings_list.append(Settings(dataset, min_instances_slice=min_instances_slice, norm=norm, class_discriminative=True))
# settings_list.append(Settings(dataset, min_instances_slice=min_instances_slice, leaf_eps=0.1, class_discriminative=True))
# settings_list.append(Settings(dataset, min_instances_slice=min_instances_slice, leaf_eps=0.2, class_discriminative=True))
# for leaf_eps in leaf_eps_list:
#     settings_list.append(Settings(dataset, min_instances_slice=min_instances_slice, norm=norm, leaf_eps=leaf_eps, class_discriminative=True))


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
#         n_epochs=n_epochs, batch_size=batch_size, learning_rate=learning_rate)
# )
# settings_list.append(
#     Settings(dataset, class_discriminative=True,
#         build_rat_spn=True, rat_spn_config=config2, rat_spn_large=True,
#         n_epochs=n_epochs, batch_size=batch_size, learning_rate=learning_rate, leaf_eps=0.1)
# )


start = time.time()
# learn_spn_test.run_test()


# spflow_mnist_test.run_test()
# spflow_rat_spn_test.run_test()
# spflow_rat_spn_test2.run_test(settings_list[0])


# Robustness evaluation run
for settings in settings_list:
    spn = None
    for leaf_eps in leaf_eps_list:
        settings.leaf_eps = leaf_eps
        spn = spflow_rob_test.run_test(settings, spn=spn)

# Script for running visualization/plotting script
# plot_rob_results.run_test(settings_list)




# analyze_data.run_test()

# Run for testing cycle detection algorithm and min/max constraints algorithm
# spflow_cycle_test.run_test()

end = time.time()
print(f'time: {end - start}')

exit()

# p0 = Product(children=[Bernoulli(p=0.7, scope=1), Bernoulli(p=0.6, scope=2)])
# p1 = Product(children=[Bernoulli(p=0.5, scope=1), Bernoulli(p=0.4, scope=2)])
# s1 = Sum(weights=[0.3, 0.7], children=[p0, p1])
# p2 = Product(children=[Bernoulli(p=0.8, scope=0), s1])
# p3 = Product(children=[Bernoulli(p=0.8, scope=0), Bernoulli(p=0.7, scope=1)])
# p4 = Product(children=[p3, Bernoulli(p=0.6, scope=2)])
# spn = Sum(weights=[0.4, 0.6], children=[p2, p4])

# print(spn)
# print()

# x = [sample(spn, np.array([np.nan, np.nan, np.nan]), RandomState(i)) for i in range(100)]
# print(sum(x) / len(x))

# train_data = [[1, 1, 1]]

# epochs = 1000
# for e in range(epochs):
#     for d in train_data:
#         ll = spn.value(d, ll=True)
#         gradient_backward(spn)
#         sgd(spn, lr=0.01, data=d)
#     if e % 100 == 0:
#         print(e, ll)


# x = [sample(spn, np.array([np.nan, np.nan, np.nan]), RandomState(i)) for i in range(100)]
# print(sum(x) / len(x))

# print()
# print(spn)

# # exit()





# print(spn.value(evidence=[np.nan, np.nan, np.nan]))
# print(spn.value(evidence=[0, np.nan, np.nan]))
# print(spn.value(evidence=[1, np.nan, np.nan]))
# print(spn.value(evidence=[np.nan, 0, np.nan]))
# print(spn.value(evidence=[np.nan, 1, np.nan]))
# print(spn.value(evidence=[np.nan, np.nan, 0]))
# print(spn.value(evidence=[np.nan, np.nan, 1]))
# print()

# # The likelihood of all possible options of evidence should be (roughly) equal to 1
# x = 0
# x += spn.value(evidence=[0, 0, 0])
# x += spn.value(evidence=[0, 0, 1])
# x += spn.value(evidence=[0, 1, 0])
# x += spn.value(evidence=[0, 1, 1])
# x += spn.value(evidence=[1, 0, 0])
# x += spn.value(evidence=[1, 0, 1])
# x += spn.value(evidence=[1, 1, 0])
# x += spn.value(evidence=[1, 1, 1])
# print(x)
# print()

# # Compare the log likelihood to that of SPFlow's documentation
# print('Computing log likelihood')
# ll = spn.value(evidence=[1, 0, 1], ll=True)
# print(ll, np.exp(ll)) # Should be [[-1.90730501]] [[0.14848]]
# print()

# # Compare the log likelihood of a marginalization to that of SPFlow's documentation
# print('Marginal inference')
# ll_mar = spn.value(evidence=[np.nan, 0, 1], ll=True)
# print(ll_mar, np.exp(ll_mar)) # Should be [[-1.68416146]] [[0.1856]]
# print()


# print(spn.value(evidence=[np.nan, np.nan, 1]))


# print(spn == spn)
# print(spn == p4)
# print(spn == s1)

# print()
# print(get_nodes(spn))

# print(get_topological_order_layers(spn))

# # def gen_sample(i):
# #     s = sample(spn, np.array([np.nan, 0, 0]), RandomState(i))
# #     print('sample', s)

# # for i in range(10):
# #     gen_sample(i)

# # print()

# # s = sample(spn, np.array([np.nan, np.nan, np.nan]), RandomState(123))
# # print('sample', s)

# # s = sample(spn, np.array([np.nan, np.nan, np.nan]), RandomState(123))
# # print('sample', s)

# print('\n\n')


# spn2 = Sum(weights=[0.5, 0.5], children=[Bernoulli(1, scope=0), Bernoulli(0, scope=0)])
# # spn2 = Bernoulli(0.5, scope=0)

# print(spn2)
# print()

# ll_mar = spn2.value(evidence=[0], ll=True)
# print('likelihood of 0: ', ll_mar, np.exp(ll_mar))
# ll_mar = spn2.value(evidence=[1], ll=True)
# print('likelihood of 1: ', ll_mar, np.exp(ll_mar))


# print()
# x = [sample(spn2, np.array([np.nan]), RandomState(i))[0] for i in range(100)]
# print(sum(x) / len(x))
# print()

# # Generate 100 data samples
# np.random.seed(1)
# data = np.random.binomial(1, p=0.2, size=100)
# test = np.random.binomial(1, p=0.2, size=20)
# np.random.seed(None)

# print('data p: ', sum(data) / len(data))

# print(spn2)

# print('\n')

# iterations = 0 # 100
# # Perform iterations
# for i in range(iterations):

#     # Train dataset
#     for di, d in enumerate(data):
#         ll = spn2.value(evidence=np.array([d]), ll=True)
#         # if di % 25 == 0:
#         #     print('\t', di, ll, np.exp(ll))
#         gradient_backward(spn2)
#         sgd(spn2, lr=0.01 / (i+1))
    
#     # Test dataset
#     test_ll = 0
#     for t in test:
#         ll = spn2.value(evidence=np.array([t]), ll=True)
#         test_ll += ll
#     print(f'Epoch {i}', 'test ll: ', test_ll / len(test))
#     x = [sample(spn2, np.array([np.nan]), RandomState(i))[0] for i in range(100)]
#     print(sum(x) / len(x))

# print()
# x = [sample(spn2, np.array([np.nan]), RandomState(i))[0] for i in range(100)]
# print(sum(x) / len(x))
# print()

# print(spn2)


# exit()

import matplotlib.pyplot as plt
import networkx as nx

# Test code for generating dense spn
rvs = [0,1,2,3,4,5,6]
# rvs = [0,1]
region_graph = random_region_graph(rvs, depth=2, repetitions=2)

print(region_graph)

constructed_spn = region_graph.to_spn(c=1, s=2, i=3, leaf_types=[Bernoulli] * 7)

add_ids_to_spn(constructed_spn)

initialize_weights(constructed_spn)

print(constructed_spn)


G = nx.Graph()

# Add all nodes
for layer in get_topological_order_layers(constructed_spn):
    for n in layer:
        if isinstance(n, Leaf):
            G.add_node(n.id, color='green')
        elif isinstance(n, Sum):
            G.add_node(n.id, color='red')
        elif isinstance(n, Product):
            G.add_node(n.id, color='steelblue')
        else:
            G.add_node(n.id, color='gray')

# Add all edges
for layer in get_topological_order_layers(constructed_spn):
    for n in layer:
        for c in n.children:
            G.add_edge(n.id, c.id)




fig, ax = plt.subplots(figsize=(20,20))

color_map = []
for node in G:
    color_map.append(G.nodes[node]['color'])

nx.draw(G, with_labels=True, font_weight='bold', node_color=color_map)

fig.savefig('network.png')

leaves = get_topological_order_layers(constructed_spn)[-1]

for leaf in leaves:
    print(f'leaf {leaf.id} - scope: {leaf.scope}')


exit()


#####################
# SPN Learning test #
#####################

def test_xor(spn, data):
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

# Train the XOR table
rvs = [0,1,2]
data = [
    [0,0,0],
    [1,1,0],
    [0,1,1],
    [1,0,1],
]
# train_data = np.array(data)[:,:2]
# rvs = [0,1]


print()
print('Learning SPN on XOR table')

# xor_spn = learn_spn(data, rvs, epochs=100, lr=0.0001)

epochs = 40
lr = 0.8
xor_spn_false = learn_spn(data[:2], rvs, epochs=epochs, lr=lr, leaf_types=[Bernoulli, Bernoulli, Bernoulli])
xor_spn_true = learn_spn(data[2:], rvs, epochs=epochs, lr=lr, leaf_types=[Bernoulli, Bernoulli, Bernoulli])
xor_spn = Sum(weights=[0.5, 0.5], children=[xor_spn_false, xor_spn_true])

print()
print('XOR SPN:')
# print(xor_spn)

for i in range(1):
    print(f'test {i}')
    test_xor(xor_spn, data)

# print(data)
