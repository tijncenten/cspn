from spn.algorithms.layerwise import layers
from spn.algorithms.Validity import is_valid
from spn.structure.Base import Sum, Product
from spn.structure.Base import assign_ids, rebuild_scopes_bottom_up
from spn.structure.leaves.parametric.Parametric import Categorical, Gaussian

from algorithms.torch.class_discriminative_layer import ClassDiscriminativeLayer

import numpy as np
import torch
from torch import nn
import itertools

# Converts a PyTorch layerwise SPN to a SPFlow object-oriented SPN
def layerwise_to_simple_spn(model, ncat, debug=False, leaf=False, test=False, rat_spn=False, config=None):
    spn = None
    if isinstance(model, ClassDiscriminativeLayer):
        spn = class_discriminative_layer_to_simple(model, ncat, debug=debug)
    elif leaf:
        spn = Sum()
        children = leaf_layer_to_simple(model, config, debug=debug)
        children = [n for feature_nodes in children for channel_nodes in feature_nodes for n in channel_nodes]
        spn.children.extend(children)
        spn.weights.extend([1.0/len(children) for i in range(len(children))])
    elif test:
        spn = Sum()
        layer_nodes = rat_spn_layer_to_simple(model, ncat, debug=debug)
        # layer_nodes = leaf_layer_to_simple(model._leaf, config)
        # layer_nodes = inner_layer_to_simple(model._inner_layers[0], layer_nodes, model, ncat)
        # layer_nodes = inner_layer_to_simple(model._inner_layers[1], layer_nodes, model, ncat)
        # layer_nodes = inner_layer_to_simple(model._inner_layers[2], layer_nodes, model, ncat)
        children = [n for feature_nodes in layer_nodes for channel_nodes in feature_nodes for n in channel_nodes]
        spn.children.extend(children)
        spn.weights.extend([1.0/len(children) for i in range(len(children))])
    elif rat_spn:
        spn = Sum()
        child = rat_spn_layer_to_simple(model, ncat, debug=debug)
        spn.children.append(child)
        spn.weights.append(1.0)
    else:
        raise NotImplementedError()

    assign_ids(spn)
    rebuild_scopes_bottom_up(spn)

    assert is_valid(spn)
    return spn


def class_discriminative_layer_to_simple(model, ncat, debug=False):
    # First convert the class discriminative root node
    spn = Sum()
    # Assign the weights
    weights = model.weights.data.numpy()
    scaled = np.exp(weights - np.max(weights))
    weights = scaled / np.sum(scaled)
    spn.weights.extend(weights)

    label_idx = len(ncat) - 1

    for class_id, rat_spn in enumerate(model.class_spns):
        # For each child spn of the root, convert to simple spn
        prod = Product()

        # Add the class label node to the product node
        k = len(model.class_spns)
        p = np.zeros(k)
        p[class_id] = 1
        class_label_node = Categorical(p=p, scope=label_idx)
        prod.children.append(class_label_node)

        # Add the RAT-SPN branch to the product node
        simple_rat_spn = rat_spn_layer_to_simple(rat_spn, ncat, debug=debug)
        prod.children.append(simple_rat_spn)

        # Add the product node to the root node
        spn.children.append(prod)


    return spn


def rat_spn_layer_to_simple(model, ncat, debug=False):
    # Make use of the variables:
    # - model._leaf (IndependentMultivariate layer)
    # - model._inner_layers (Both Sum and CrossProduct layers)
    # - model.root (A Sum layer)

    leaf_nodes = leaf_layer_to_simple(model._leaf, model.config, debug=debug)
    
    # Inner layer nodes
    layer_nodes = leaf_nodes
    for inner_layer in model._inner_layers:
        layer_nodes = inner_layer_to_simple(inner_layer, layer_nodes, model, ncat, debug=debug)

    # Root node (Sum layer)
    assert len(layer_nodes) == 1 # number of features should be 1 at this point

    # Merge results from the different repetitions into the channel dimension
    layer_nodes = [[[node] for channel_nodes in feature_nodes for node in channel_nodes] for feature_nodes in layer_nodes]

    if debug:
        print(f'({len(layer_nodes)},{len(layer_nodes[0])},{len(layer_nodes[0][0])})')

    root_nodes = inner_layer_to_simple(model.root, layer_nodes, model, ncat, debug=debug)

    root = root_nodes[0][0][0]

    return root


def leaf_layer_to_simple(model, config, debug=False):
    # Leaf nodes
    # in_features = F (different scopes / Product layer)
    # out_channels = I (same scope / Sum layer)
    # num_repetitions = R (disconnected until root)
    # Cardinality is the size of the region in the last partition
    in_features = config.F
    out_channels = config.I
    num_repetitions = config.R

    means = model.base_leaf.means.data.numpy()[0]
    stds = model.base_leaf.stds.data.numpy()[0]
    min_sigma = model.base_leaf.min_sigma
    max_sigma = model.base_leaf.max_sigma
    if min_sigma < max_sigma:
        sigma_ratio = 1 / (1 + np.exp(-stds))
        sigma = min_sigma + (max_sigma - min_sigma) * sigma_ratio
    else:
        sigma = 1.0

    # Create leaf nodes with dimensions (F, I, R)
    leaf_nodes = []
    for f in range(in_features):
        feature_nodes = []
        for i in range(out_channels):
            channel_nodes = []
            for r in range(num_repetitions):
                mean = means[f][i][r]
                stdev = sigma[f][i][r] if min_sigma < max_sigma else sigma
                l = Gaussian(mean=mean, stdev=np.sqrt(stdev), scope=f)
                channel_nodes.append(l)
            feature_nodes.append(channel_nodes)
        leaf_nodes.append(feature_nodes)

    if debug:
        print(f'({len(leaf_nodes)},{len(leaf_nodes[0])},{len(leaf_nodes[0][0])})')

    # Merge leaf nodes with product nodes
    # Each product node has #cardinality children (with padding of 0 at the end)
    # Therefore, d_out = d // cardinality product nodes should be created
    cardinality = model.prod.cardinality
    d_out = in_features // cardinality

    product_nodes = []
    for d in range(d_out):
        feature_nodes = []
        for i in range(out_channels):
            channel_nodes = []
            for r in range(num_repetitions):
                p = Product()
                for c in range(cardinality):
                    f = d * cardinality + c
                    if f >= in_features:
                        break
                    l = leaf_nodes[f][i][r]
                    p.children.append(l)
                channel_nodes.append(p)
            feature_nodes.append(channel_nodes)
        product_nodes.append(feature_nodes)

    if debug:
        print(f'({len(product_nodes)},{len(product_nodes[0])},{len(product_nodes[0][0])})')


    return product_nodes
    
def inner_layer_to_simple(inner_layer, input_nodes, model, ncat, debug=False):
    layer_nodes = []
    if isinstance(inner_layer, layers.CrossProduct):
        in_features = inner_layer.in_features
        num_repetitions = inner_layer.num_repetitions
        in_channels = inner_layer.in_channels
        cardinality = 2 # Fixed
        assert cardinality == inner_layer.cardinality
        scopes = inner_layer._scopes

        d_out = in_features // cardinality

        # TODO: Possibly pad to next power of 2



        layer_nodes = []
        for d in range(d_out):
            feature_nodes = []
            for co in range(in_channels ** 2):
                # co = cl * 2 + cr
                cl = co // in_channels
                cr = co % in_channels
                channel_nodes = []
                for r in range(num_repetitions):
                    p = Product()
                    # Left child
                    f_left = scopes[0][d]
                    n_left = input_nodes[f_left][cl][r]
                    p.children.append(n_left)
                    # Right child
                    f_right = scopes[1][d]
                    if f_right < len(input_nodes):
                        n_right = input_nodes[f_right][cr][r]
                        p.children.append(n_right)
                    # Add to channel nodes
                    channel_nodes.append(p)
                feature_nodes.append(channel_nodes)
            layer_nodes.append(feature_nodes)

        if debug:
            print(f'({len(layer_nodes)},{len(layer_nodes[0])},{len(layer_nodes[0][0])})')
        

    elif isinstance(inner_layer, layers.Sum):
        in_features = inner_layer.in_features
        num_repetitions = inner_layer.num_repetitions
        in_channels = inner_layer.in_channels
        out_channels = inner_layer.out_channels

        weights = inner_layer.weights.data.numpy()

        # scaled = np.exp(weights - np.max(weights))
        # weights = scaled / np.sum(scaled)
        # spn.weights.extend(weights)

        layer_nodes = []
        assert in_features == len(input_nodes)
        for d in range(in_features):
            feature_nodes = []
            for oc in range(out_channels):
                channel_nodes = []
                for r in range(num_repetitions):
                    s = Sum()
                    # Add the weights to the sum node
                    w = weights[d, :, oc, r]
                    scaled = np.exp(w - np.max(w))
                    w = scaled / np.sum(scaled)
                    s.weights.extend(w)
                    # Add the child nodes to the sum node
                    for ic in range(in_channels):
                        s.children.append(input_nodes[d][ic][r])

                    channel_nodes.append(s)
                feature_nodes.append(channel_nodes)
            layer_nodes.append(feature_nodes)

        if debug:
            print(f'({len(layer_nodes)},{len(layer_nodes[0])},{len(layer_nodes[0][0])})')

    else:
        raise NotImplementedError()
    
    return layer_nodes
