
import numpy as np
import torch

from spn.experiments.RandomSPNs_layerwise.distributions import RatNormal
from spn.experiments.RandomSPNs_layerwise.rat_spn import RatSpn, RatSpnConfig

from algorithms.torch.class_discriminative_layer import CustomRatSpn
from algorithms.torch.layerwise_to_simple import layerwise_to_simple_spn
from prep import get_data
from experiments.settings import Settings

from algorithms.graphics import plot_labeled_spn

def run_test(settings=None):
    if settings is None:
        raise ValueError('settings cannot be None')

    dataset_name = settings.dataset_name

    np.random.seed(0)
    data, ncat = get_data(dataset_name)
    
    config = settings.rat_spn_config
    if config is None:
        raise ValueError('RAT-SPN config cannot be None')
    
    config.F = len(ncat) - 1

    spn = CustomRatSpn(config)

    simple_spn = layerwise_to_simple_spn(spn, ncat, rat_spn=True)

    plot_labeled_spn(simple_spn,
        f'src/report-vis/rat-spn-example.png',
        large=False
    )
