from spn.structure.Base import Sum, Product
from spn.structure.Base import assign_ids, rebuild_scopes_bottom_up
from spn.structure.leaves.parametric.Parametric import create_parametric_leaf

import itertools
import numpy as np


# The RAT-SPN algorithm as described in the following paper
# src: https://arxiv.org/pdf/1806.01910.pdf
def learn_rat_spn(
    data,
    ds_context,
    depth=2,
    repetitions=2,
    c=1,
    s=2,
    i=3,
):
    variables = list(range(data.shape[1]))
    # First construct a region graph
    region_graph = random_region_graph(variables, depth=depth, repetitions=repetitions)
    # Convert the region graph to an spn
    spn = region_graph.to_spn(data, ds_context, c=c, s=s, i=i)

    assign_ids(spn)
    rebuild_scopes_bottom_up(spn)

    return spn


class RegionNode():
    # Initialize a region graph node by storing the region {r}
    def __init__(self, r, root=False):
        self.r = sorted(r)
        self.spn_nodes = []
        self.partitions = []
        self.root = root

    # Method for adding a partition to this region node
    # A partition is a tuple of different RegionNode objects
    def add_partition(self, partition):
        self.partitions.append(partition)

    # Method for converting the region node and its children to an spn recursively
    # using the parameters as described in the paper
    def to_spn(self, data, ds_context, c=1, s=2, i=3):
        if c > 1:
            raise ValueError('Multiple roots not yet implemented')

        # First, initialize the spn nodes for this region node
        if len(self.partitions) == 0:
            # Leaf node
            self.init_spn_leaves(i, ds_context, data)
        elif self.root:
            # Root node
            self.init_spn_nodes(Sum, c)
        else:
            # Internal node
            self.init_spn_nodes(Sum, s)
        
        # Next, convert the child regions in the partitions to spns
        # (recursive step)
        for r1, r2 in self.partitions:
            r1.to_spn(data, ds_context, c=c, s=s, i=i)
            r2.to_spn(data, ds_context, c=c, s=s, i=i)

        # Finally, add product nodes between the current node and its children
        # based on the created partitions
        for i, (r1, r2) in enumerate(self.partitions):
            # Cartesian product between all spn nodes of both regions
            for n1, n2 in itertools.product(r1.spn_nodes, r2.spn_nodes):
                p = Product(children=[n1, n2])
                cnt = len(r1.spn_nodes) * len(r2.spn_nodes) * len(self.partitions)
                for n in self.spn_nodes:
                    n.children.append(p)
                    if isinstance(n, Sum):
                        n.weights.append(1. / cnt)

        if c == 1:
            return self.spn_nodes[0]
        return self.spn_nodes

    # Method for initializing the spn nodes of this region graph node
    def init_spn_nodes(self, node_type, number):
        self.spn_nodes = [node_type() for i in range(number)]

    # Method for initializing the spn nodes of this region graph node as a leaf
    def init_spn_leaves(self, number, ds_context, data):
        if ds_context is None:
            raise ValueError('No Context provided')
        # For every rv in r, a leaf node is created
        # leaf_nodes = [Bernoulli(p=0.5, scope=s) for s in self.r]
        # leaf_nodes = [leaf_types[s](scope=s) for s in self.r]
        # leaf_nodes = [leaf(scope=s) for leaf, s in zip(ds_context.get_parametric_types_by_scope(self.r), self.r)]
        leaf_nodes = [create_parametric_leaf(data[:,s].reshape(-1, 1), ds_context, [s]) for s in self.r]
        # leaf_nodes = [Sum(weights=[0.5, 0.5], children=[Bernoulli(p=1, scope=s), Bernoulli(p=0, scope=s)]) for s in self.r]
        # Create {number} different product nodes which all have the leaves as children
        self.spn_nodes = [Product(children=leaf_nodes) for i in range(number)]


# Method for creating a random region graph
def random_region_graph(variables, depth, repetitions):
    node = RegionNode(variables, root=True)
    # Create {repetitions} different splits of the original rvs {variables}
    for r in range(repetitions):
        split(node, depth)
    return node

def split(region, depth):
    # Create two random balanced partitions
    r_region = np.random.permutation(region.r)
    P = r1, r2 = RegionNode(r_region[:len(region.r)//2]), RegionNode(r_region[len(region.r)//2:])
    # Insert P in region
    region.add_partition(P)
    if depth > 1:
        if len(r1.r) > 1: split(r1, depth-1)
        if len(r2.r) > 1: split(r2, depth-1)
