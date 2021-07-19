
from spn.algorithms.Validity import is_valid
from spn.structure.leaves.parametric.Parametric import Categorical, Gaussian
from spn.structure.Base import Context, Sum, Product

from algorithms.graphics import plot_labeled_spn
from algorithms.learning import learn_class_discriminative
from algorithms.nodes import CollapsedNode

import numpy as np
import matplotlib.pyplot as plt

def get_small_class_discriminative_structure(**kwargs):
    def build_simple_spn(*args, **kwargs):
        # l1 = Product(children=[])
        # l2 = Product(children=[])
        # p = Sum(children=[l1, l2])
        p = Sum(children=[])
        p = CollapsedNode(scope=0)
        return p
        # return get_small_cycle_structure()

    train_data = np.array([[0,0,0], [0,1,1], [1,0,2], [1,1,0]])
    p_types = [Categorical, Categorical, Categorical]

    spn = learn_class_discriminative(train_data,
        Context(parametric_types=p_types).add_domains(train_data),
        build_simple_spn, 2)

    return spn

spn = get_small_class_discriminative_structure()
assert is_valid(spn)

large_plot = True
scope_letters = [None, None, 'c']

plot_labeled_spn(spn,
    f'src/report-vis/cd-structure-example.png',
    large=large_plot,
    scope_letters=scope_letters
)
plot_labeled_spn(spn,
    f'src/report-vis/cd-structure-example.pdf',
    large=large_plot,
    scope_letters=scope_letters
)


