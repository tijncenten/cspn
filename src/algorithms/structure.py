from spn.structure.Base import Sum, Product, Leaf, get_nodes_by_type, get_topological_order_layers
from collections import OrderedDict

def is_spn_tree(node):
    nodes = get_nodes_by_type(node)
    parents = OrderedDict({node: []})

    for n in nodes:
        if not isinstance(n, Leaf):
            for c in n.children:
                parent_list = parents.get(c, None)
                if parent_list is None:
                    parents[c] = parent_list = []
                parent_list.append(n)
    

    is_tree = True

    for n, ps in parents.items():
        if len(ps) > 1:
            # print(f'node {n} has {len(ps)} parents')
            is_tree = False
    
    return is_tree


def compute_tree_nodes(node):
    nodes = get_nodes_by_type(node)
    parents = OrderedDict({node: []})

    for n in nodes:
        if not isinstance(n, Leaf):
            for c in n.children:
                parent_list = parents.get(c, None)
                if parent_list is None:
                    parents[c] = parent_list = []
                parent_list.append(n)

    for n, ps in parents.items():
        if len(ps) > 1:
            n._tree_node = False
        else:
            n._tree_node = True


def compute_node_depth(node, depth=0):
    if hasattr(node, '_depth'):
        node._depth = min(depth, node._depth)
    else:
        node._depth = depth
    if isinstance(node, Leaf):
        return
    for c in node.children:
        compute_node_depth(c, depth + 1)


def get_number_of_parameters(spn):
    sum_nodes = get_nodes_by_type(spn, Sum)
    nr_params = 0
    for n in sum_nodes:
        nr_params += len(n.weights)
    return nr_params


def get_structure_cycles(node):
    cycles = []

    cycle_opens = []

    nodes = get_nodes_by_type(node)
    parents = OrderedDict({node: []})

    for n in nodes:
        if not isinstance(n, Leaf):
            for c in n.children:
                parent_list = parents.get(c, None)
                if parent_list is None:
                    parents[c] = parent_list = []
                parent_list.append(n)
    
    layers = get_topological_order_layers(node)
    layer_index = {}
    for i, layer in enumerate(layers):
        for n in layer:
            if len(parents[n]) > 1:
                # This node has multiple parents, so opens a cycle
                cycle_opens.append(n)
            layer_index[n] = i

    for layer in layers:
        pass

    # For each cycle open, find the complete cycle
    for n in cycle_opens:
        cycle_front = [p for p in parents[n]]
        cycle_nodes_idx = set([n.id])

        while len(cycle_front) > 1:
            # Sort the cycle front on lowest layer first
            cycle_front.sort(key=lambda p: layer_index[p])
            # Process the first node in the front; and add to the cycle nodes
            choice = cycle_front.pop(0)
            cycle_nodes_idx.add(choice.id)
            # Add the parents of the node to the cycle front
            choice_p = parents[choice]
            for p in choice_p:
                if not p in cycle_front:
                    cycle_front.append(p)

        cycle_close = cycle_front[0]
        cycle_nodes_idx.add(cycle_close.id)
        cycles.append((n, cycle_close, cycle_nodes_idx))

    print('done')


def check_tractable_robustness(node, class_var=None):
    assert class_var != None
    nodes = get_nodes_by_type(node)
    parents = OrderedDict({node: []})

    for n in nodes:
        if not isinstance(n, Leaf):
            for c in n.children:
                parent_list = parents.get(c, None)
                if parent_list is None:
                    parents[c] = parent_list = []
                parent_list.append(n)

    layers = get_topological_order_layers(node)
    constraints = {}
    for n in nodes:
        constraints[n] = {}
    
    for root in layers[-1]:
        constraints[root] = {'MIN': []}

    def merge_constraints(child_constraints, node_constraints, condition_node=None):
        constraints = child_constraints.copy()
        if 'MIN' in node_constraints:
            constraints.setdefault('MIN', []).extend(node_constraints['MIN'])
        if 'MAX' in node_constraints:
            constraints.setdefault('MAX', []).extend(node_constraints['MAX'])
        if condition_node != None:
            condition_pos = f'{condition_node.id}>=0'
            condition_neg = f'{condition_node.id}<0'
            assert not ('MIN' in node_constraints and 'MAX' in node_constraints)
            if 'MIN' in node_constraints:
                constraints.setdefault('MIN', []).append(condition_pos)
                constraints.setdefault('MAX', []).append(condition_neg)
            if 'MAX' in node_constraints:
                constraints.setdefault('MAX', []).append(condition_pos)
                constraints.setdefault('MIN', []).append(condition_neg)
        return constraints

    for layer in reversed(layers):
        for n in layer:
            if isinstance(n, Leaf):
                continue
            if isinstance(n, Product) and class_var in n.scope:
                # min/max case
                # Find the child with class_var in scope
                class_node = [c for c in n.children if class_var in c.scope]
                assert len(class_node) == 1
                class_node = class_node[0]

                # Propagate the constraints
                for c in n.children:
                    if c == class_node:
                        # Propagate the parent constraints
                        constraints[c] = merge_constraints(constraints[c], constraints[n])
                    else:
                        # Propagate the class_var constraints
                        constraints[c] = merge_constraints(constraints[c], constraints[n], condition_node=class_node)
            else:
                for c in n.children:
                    constraints[c] = merge_constraints(constraints[c], constraints[n])

    print('done')

