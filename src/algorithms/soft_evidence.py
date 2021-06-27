import numpy as np

from spn.algorithms.Inference import add_node_likelihood, leaf_marginalized_likelihood
from spn.structure.leaves.parametric.Parametric import Gaussian, Categorical
from spn.structure.leaves.parametric.utils import get_scipy_obj_params

def compute_leaf_value(node, **kwargs):
    if isinstance(node, Gaussian):
        return continuous_soft_evidence(node, **kwargs)
    else:
        raise Exception(f'node type {node} not implemented')


def continuous_soft_evidence(node, data=None, leaf_eps=0.1, dtype=np.float64, **kwargs):
    probs, marg_ids, observations = leaf_marginalized_likelihood(node, data, dtype, log_space=True)
    scipy_obj, params = get_scipy_obj_params(node)
    offset = leaf_eps / 2 * params['scale']
    obs_max = observations + offset
    obs_min = observations - offset
    probs[~marg_ids] = np.log(scipy_obj.cdf(obs_max, **params) - scipy_obj.cdf(obs_min, **params))
    return probs
