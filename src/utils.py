import numpy as np

from node import Sum, Product, Leaf, Bernoulli, Categorical

# Method for retrieving all nodes in a list, from the root node down
def get_nodes(node):
    if len(node.children) == 0:
        return [node]
    nodes = [node]
    for c in node.children:
        nodes.extend(get_nodes(c))
    return nodes


# Method for constructing topological layers of the spn, starting from the root node
# This is done by adding all nodes, for which all parents have already been added
# to some previous layer, to the current layer
def get_topological_order_layers(root):
    layers = []

    nodes_added = []
    all_nodes = get_nodes(root)
    nr_added = 0

    # Add the root node as the first layer
    layers.append([root])
    nodes_added.append(root)
    nr_added += 1

    # Repeatedly add all nodes, for which all parent nodes have been added
    while True:
        # Add consecutive layers
        layer = []

        # For all nodes in the previous layer
        for n in layers[-1]:
            # Check all candidate nodes, being children of n
            for c in n.children:
                # Check if all parents of c are already added AND c is not yet added
                if all(p in nodes_added for p in c.parents) and not c in nodes_added:
                    # Add c to the current layer
                    layer.append(c)
                    nodes_added.append(c)
        # Stop the algorithm, when no new nodes have been added to the current layer
        if len(layer) == 0:
            break

        layers.append(layer)
        nr_added += len(layer)
    return layers


# Method for assigning unique ids to an already constructed spn
def add_ids_to_spn(root):
    if root.id != None:
        return False
    id = 0
    for layer in get_topological_order_layers(root):
        for n in layer:
            n.set_index(id)
            id += 1
    return True


def eval_spn_bottom_up(root, eval_functions, results=None):
    pass


# Method for generating a sample from the spn, using the input_data as conditioning
def sample(root, input_data, rand_gen=None, results=None):
    # Make a copy of the input data
    data = np.array(input_data)
    # compute the log likelihoods
    root.value(evidence=data, ll=True)

    if results is None:
        results = {}
    
    root_result = [0]
    results[root] = [root_result]

    # Iterate over the spn top down over all layers
    for layer in get_topological_order_layers(root):
        for n in layer:
            # Get the parameter for the current node n
            param = results[n]
            # Retrieve result from node n
            # If n is a leaf node, data can be updated if the rv value is missing
            result = n.sample(param, data=data, rand_gen=rand_gen)

            # If there is a result, add it to the results lookup for a next iteration
            if result is not None and not isinstance(n, Leaf):
                for child, param in result.items():
                    if child not in results:
                        results[child] = []
                    results[child].append(param)

    # The data array contains the sample
    return data


# Method for generating a sample from the spn using mpe, using the input_data as conditioning
def mpe(root, input_data, rand_gen=None, results=None):
    # Make a copy of the input data
    data = np.array(input_data)
    # compute the log likelihoods
    root.value(evidence=data, ll=True)

    if results is None:
        results = {}
    
    root_result = [0]
    results[root] = [root_result]

    # Iterate over the spn top down over all layers
    for layer in get_topological_order_layers(root):
        for n in layer:
            # Get the parameter for the current node n
            param = results[n]
            # Retrieve result from node n
            # If n is a leaf node, data can be updated if the rv value is missing
            result = n.mpe(param, data=data, rand_gen=rand_gen)

            # If there is a result, add it to the results lookup for a next iteration
            if result is not None and not isinstance(n, Leaf):
                for child, param in result.items():
                    if child not in results:
                        results[child] = []
                    results[child].append(param)

    # The data array contains the sample
    return data


def gradient_backward(root):
    results = {}

    root_result = 0
    results[root] = [root_result]

    for layer in get_topological_order_layers(root):
        for n in layer:
            param = results[n]

            result = n.gradient(param)

            if result is not None and not isinstance(n, Leaf):
                for child, param in result.items():
                    if child not in results:
                        results[child] = []
                    results[child].append(param)
                    # if len(results[child]) > 1:
                    #     print(f'more results: {child}')


def sgd(root, lr=0.05, **kwargs):
    for layer in get_topological_order_layers(root):
        for n in layer:
            n.sgd(lr=lr, **kwargs)
