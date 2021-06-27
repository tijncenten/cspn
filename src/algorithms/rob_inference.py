from spn.structure.Base import Product, Sum, Leaf, eval_spn_bottom_up
from spn.algorithms import Inference
from algorithms import soft_evidence

import numpy as np
from scipy.special import logsumexp

# Methods for computing the econtaminated likelihood with robustness
# The log_likelihood implementation of SPFlow and robustness examples
# of GeFs have been used as inspiration
# source: https://github.com/SPFlow/SPFlow/blob/master/src/spn/algorithms/Inference.py
# source: https://github.com/tijncenten/GeFs/blob/master/gefs/nodes.py

def prod_rob_log_likelihood(node, children, dtype=np.float64, **kwargs):
    llchildren = np.concatenate(children, axis=2)
    assert llchildren.dtype == dtype
    pll = np.sum(llchildren, axis=2).reshape(2, -1, 1)
    pll[np.isinf(pll)] = np.finfo(pll.dtype).min

    return pll

def prod_rob_likelihood(node, children, dtype=np.float64, **kwargs):
    llchildren = np.concatenate(children, axis=2)
    assert llchildren.dtype == dtype
    return np.prod(llchildren, axis=2).reshape(2, -1, 1)

def sum_rob_log_likelihood(node, children, dtype=np.float64, eps=0.1, **kwargs):
    llchildren = np.concatenate(children, axis=2)
    assert llchildren.dtype == dtype

    assert np.isclose(np.sum(node.weights), 1.0), "unnormalized weights {} for node {}".format(node.weights, node)

    w_min = econtaminate(node.weights, llchildren[0], eps, False)
    w_max = econtaminate(node.weights, llchildren[1], eps, True)

    sll_min = logsumexp(llchildren[0], b=w_min, axis=1).reshape(-1, 1)
    sll_max = logsumexp(llchildren[1], b=w_max, axis=1).reshape(-1, 1)

    return np.array([sll_min, sll_max])

# Implementation for e-contamination, based on the fractional knapsack problem
# source: https://github.com/tijncenten/GeFs/blob/master/gefs/nodes.py
def econtaminate(vec, logprs, eps, ismax):
    econt = np.asarray(vec) * (1 - eps)
    # Create an e-contaminated array of vec, for each instance of logprs
    econt = np.repeat(np.array([econt]), logprs.shape[0], axis=0)
    for j in range(econt.shape[0]):
        room = 1 - np.sum(econt[j])
        if ismax:
            order = np.argsort(-1*logprs[j])
        else:
            order = np.argsort(logprs[j])
        for i in order:
            if room > eps:
                econt[j, i] = econt[j, i] + eps
                room -= eps
            else:
                econt[j, i] = econt[j, i] + room
                break
    return econt

def sum_rob_likelihood(node, children, dtype=np.float64, eps=0.1, **kwargs):
    raise ValueError('Not implemented yet')
    llchildren = np.concatenate(children, axis=1)
    assert llchildren.dtype == dtype

    assert np.isclose(np.sum(node.weights), 1.0), "unnormalized weights {} for node {}".format(node.weights, node)

    b = np.array(node.weights, dtype=dtype)

    return np.dot(llchildren, b).reshape(-1, 1)

def _d(f):
    def func(*args, leaf_eps=None, **kwargs):
        if leaf_eps is not None:
            raise NotImplementedError()
        val = f(*args, **kwargs)
        return np.array([val, val])
    return func

_node_rob_log_likelihood = {k:_d(v) for k,v in Inference._node_log_likelihood.items()}
_node_rob_likelihood = {k:_d(v) for k,v in Inference._node_likelihood.items()}

_node_rob_log_likelihood[Sum] = sum_rob_log_likelihood
_node_rob_log_likelihood[Product] = prod_rob_log_likelihood
_node_rob_likelihood[Sum] = sum_rob_likelihood
_node_rob_likelihood[Product] = prod_rob_likelihood

def rob_likelihood(node, data, dtype=np.float64, node_rob_likelihood=_node_rob_likelihood, lls_matrix=None, debug=False, **kwargs):
    '''
    Computes the robustness likelihood, using e-contamination on the sum node weights

    Parameters
    ----------
    node: Node object
        The root node of the SPN
    data: dict
        A numpy array of size (N,C) with N being instances and C being variables
    eps: float or None (default 0.1)
        The epsilon value for the e-contamination
    
    Returns
    -------
    A numpy array of size (2, N, 1) with the min and max likelihood as first dimension
    '''
    return Inference.likelihood(node, data, dtype=dtype, node_likelihood=node_rob_likelihood, lls_matrix=lls_matrix, debug=debug, **kwargs)

def rob_log_likelihood(node, data, dtype=np.float64, node_rob_log_likelihood=_node_rob_log_likelihood, lls_matrix=None, debug=False, **kwargs):
    '''
    Computes the robustness log likelihood, using e-contamination on the sum node weights

    Parameters
    ----------
    node: Node object
        The root node of the SPN
    data: dict
        A numpy array of size (N,C) with N being instances and C being variables
    eps: float or None (default 0.1)
        The epsilon value for the e-contamination
    
    Returns
    -------
    A numpy array of size (2, N, 1) with the min and max log likelihood as first dimension
    '''
    return rob_likelihood(node, data, dtype=dtype, node_rob_likelihood=node_rob_log_likelihood, lls_matrix=lls_matrix, debug=debug, **kwargs)
