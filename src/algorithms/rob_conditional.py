from spn.structure.Base import Product, Sum, Leaf, eval_spn_bottom_up
from spn.structure.leaves.parametric.Parametric import Categorical
from spn.algorithms import Inference
from algorithms.signed import Signed, signed_econtaminate
from algorithms import rob_inference
from algorithms import soft_evidence

import numpy as np
from scipy.special import logsumexp

# Methods for computing the robustness of provided instances
# which can be used to perform credal classification
# The log_likelihood implementation of SPFlow and robustness examples
# of GeFs have been used as inspiration
# source: https://github.com/SPFlow/SPFlow/blob/master/src/spn/algorithms/Inference.py
# source: https://github.com/tijncenten/GeFs/blob/master/gefs/nodes.py

def prod_rob_class_eps(node, children, dtype=np.float64, class_var=None, n_classes=None, eps=None, pred_class=None, max_depth=None, filter_tree_nodes=False, **kwargs):
    assert class_var != None and n_classes != None and eps != None and pred_class != None
    if max_depth != None:
        if not hasattr(node, '_depth'):
            raise ValueError('Node depth has not been computed')
        # If this node has depth higher than the max depth, then just evaluate the node
        if node._depth > max_depth:
            # Evaluate the node using eps = 0
            eps = 0
    if filter_tree_nodes:
        if not hasattr(node, '_tree_node'):
            raise ValueError('Node tree_node has not been computed')
        # If this node has a single parent, and all parents are part of a tree, then use eps
        # else, eps = 0
        if not node._tree_node:
            eps = 0

    if not class_var in node.scope:
        # The class variable is not in the scope of the node, so the result is equal to
        # that of the robust log likelihood
        return rob_inference.prod_rob_log_likelihood(node, children, dtype=dtype, eps=eps, **kwargs)
    # The class variable is part of the node scope
    res_min = Signed(np.array([1]), None)
    res_max = Signed(np.array([1]), None)
    res_min_cl = None
    res_max_cl = None

    # First compute the product of all non-class variables
    # and store the class variables
    for i in range(len(node.children)):
        if class_var in node.children[i].scope:
            # Store the min and max values of child k
            res_min_cl = children[i][0]
            res_max_cl = children[i][1]
        else:
            # Product is evaluated as sum in log domain for Signed values
            res_min = res_min * children[i][0][0]
            res_max = res_max * children[i][1][0]
            assert res_min.sign[0] >= 0
            assert res_max.sign[0] >= 0
    
    new_res_min = Signed(np.array([]))
    new_res_max = Signed(np.array([]))

    # Compute the conditional equation of a product node
    # based on the sign of V_k
    if res_min_cl != None and res_max_cl != None:
        for j in range(n_classes):
            # For each class, compute the result of the product node
            # Compute the min value
            if res_min_cl.sign[j] < 0:
                res_min_j = res_min_cl.get(j) * res_max
            else:
                res_min_j = res_min_cl.get(j) * res_min
            new_res_min = new_res_min.concat(res_min_j)
            # Compute the max value
            if res_max_cl.sign[j] < 0:
                res_max_j = res_max_cl.get(j) * res_min
            else:
                res_max_j = res_max_cl.get(j) * res_max
            new_res_max = new_res_max.concat(res_max_j)
        return [new_res_min, new_res_max]
    return [res_min, res_max]


def sum_rob_class_eps(node, children, dtype=np.float64, class_var=None, n_classes=None, eps=None, pred_class=None, max_depth=None, filter_tree_nodes=False, **kwargs):
    assert class_var != None and n_classes != None and eps != None and pred_class != None
    if max_depth != None:
        if not hasattr(node, '_depth'):
            raise ValueError('Node depth has not been computed')
        # If this node has depth higher than the max depth, then just evaluate the node
        if node._depth > max_depth:
            # Evaluate the node using eps = 0
            eps = 0
    if filter_tree_nodes:
        if not hasattr(node, '_tree_node'):
            raise ValueError('Node tree_node has not been computed')
        # If this node has a single parent, and all parents are part of a tree, then use eps
        # else, eps = 0
        if not node._tree_node:
            eps = 0

    if not class_var in node.scope:
        # The class variable is not in the scope of the node, so the result is equal to
        # that of the robust log likelihood
        return rob_inference.sum_rob_log_likelihood(node, children, dtype=dtype, eps=eps, **kwargs)
    # The class variable is part of the node scope
    values_min, values_max = np.zeros((len(node.children), n_classes)), np.zeros((len(node.children), n_classes))
    signs_min, signs_max = np.zeros((len(node.children), n_classes)), np.zeros((len(node.children), n_classes))
    # NOTE: All children include class_var in the scope
    # so, values for all n_classes are present
    for i in range(len(node.children)):
        values_min[i, :], signs_min[i, :] = children[i][0].value, children[i][0].sign
        values_max[i, :], signs_max[i, :] = children[i][1].value, children[i][1].sign
    
    res_min = Signed(np.zeros(n_classes), None)
    res_max = Signed(np.zeros(n_classes), None)
    for j in range(n_classes):
        # Compute the min value
        min_j = Signed(values_min[:, j], signs_min[:, j])
        w_min = signed_econtaminate(node.weights, min_j, eps, False)
        res_min_j = min_j * w_min
        res_min.insert(res_min_j.sum(), j)
        # Compute the max value
        max_j = Signed(values_max[:, j], signs_max[:, j])
        w_max = signed_econtaminate(node.weights, max_j, eps, True)
        res_max_j = max_j * w_max
        res_max.insert(res_max_j.sum(), j)
    return [res_min, res_max]


def _d(f):
    def func(node, class_var=None, n_classes=None, eps=None, pred_class=None, max_depth=None, leaf_eps=None, **kwargs):
        # Conditional expectation function if j = q (see paper)
        if class_var in node.scope:
            # Create an array of length j (num_classes) and subtract v_j from v_pred
            if isinstance(node, Categorical):
                assert len(node.p) == n_classes
                p = np.array(node.p)
                p = p[pred_class] - p
                s = Signed(p, None)
                return [s, s]
            else:
                raise ValueError(f'Node {node} not implemented as class variable')
        else:
            if leaf_eps is not None:
                val = soft_evidence.compute_leaf_value(node, leaf_eps=leaf_eps, **kwargs)
            else:
                val = f(node, **kwargs)
        return np.array([val, val])
    return func

_node_class_eps = {k:_d(v) for k,v in Inference._node_log_likelihood.items()}

_node_class_eps[Sum] = sum_rob_class_eps
_node_class_eps[Product] = prod_rob_class_eps

def rob_class_eps(spn, evi, dtype=np.float64, node_class_eps=_node_class_eps, debug=False, **kwargs):
    all_results = {}

    result = eval_spn_bottom_up(spn, node_class_eps, all_results=all_results, debug=debug, dtype=dtype, data=evi, **kwargs)

    return result

# source: https://github.com/tijncenten/GeFs/blob/master/gefs/nodes.py
def rob_class_binary_search(spn, evi, class_var, n_classes, pred_class, **kwargs):
    lower = 0
    upper = 1
    rob = (lower + upper) / 2
    it = 0
    while (lower < upper - 0.005) and (it <= 200):
        ok = True
        rob = (lower + upper) / 2
        min_values, max_values = rob_class_eps(spn, evi, class_var=class_var, n_classes=n_classes, eps=rob, pred_class=pred_class, **kwargs)
        for j in range(n_classes):
            if j != pred_class:
                if min_values.get(j).sign[0] <= 0:
                    ok = False
                    break
        if ok:
            lower = rob
        else:
            upper = rob
        it += 1
    return rob

def rob_classification(spn, evi, class_var, n_classes, pred=None, progress=False, **kwargs):
    rob = np.zeros(evi.shape[0])
    if pred is None:
        raise ValueError('Implement predictions in method')

    iterable = range(evi.shape[0])
    if progress:
        from tqdm import tqdm
        iterable = tqdm(iterable)

    for i in iterable:
        rob[i] = rob_class_binary_search(spn, evi[i:i+1], class_var, n_classes, pred[i], **kwargs)

    if pred is None:
        return pred, rob
    return rob
