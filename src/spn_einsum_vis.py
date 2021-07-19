
from spn.algorithms.Validity import is_valid
from spn.structure.leaves.parametric.Parametric import Categorical, Gaussian
from spn.structure.Base import Context, Sum, Product
from spn.structure.Base import assign_ids, rebuild_scopes_bottom_up

from algorithms.graphics import plot_labeled_spn
from algorithms.learning import learn_class_discriminative
from algorithms.nodes import CollapsedNode

import numpy as np
import matplotlib.pyplot as plt

def get_einsum_operation_structure(**kwargs):
    c1 = CollapsedNode(scope=0)
    c2 = CollapsedNode(scope=1)
    p = Product([c1, c2])
    spn = Sum(weights=[1.0], children=[p])

    assign_ids(spn)
    rebuild_scopes_bottom_up(spn)

    return spn

spn = get_einsum_operation_structure()
assert is_valid(spn)

large_plot = True
margins = 0.1

fig = plt.figure(figsize=(3,4))

plot_labeled_spn(spn,
    f'src/report-vis/spn-einsum-example.png',
    large=large_plot,
    margins=margins,
    save=False
)


fig.text(0.42, 0.7, '$\mathbf{W}$', fontsize=12)

fig.text(0.32, 0.9, '$\mathsf{S}$', fontsize=18)
fig.text(0.32, 0.48, '$\mathsf{P}$', fontsize=18)
fig.text(0.2, 0.07, '$\mathsf{N}$', fontsize=18)
fig.text(0.7, 0.07, '$\mathsf{N\'}$', fontsize=18)

fig.savefig(f'src/report-vis/spn-einsum-example.png')
fig.savefig(f'src/report-vis/spn-einsum-example.pdf')


