from spn.algorithms.Validity import is_valid
from spn.structure.leaves.parametric.Parametric import Categorical, Gaussian
from spn.structure.Base import Context, Sum, Product
from spn.structure.Base import assign_ids, rebuild_scopes_bottom_up

from algorithms.graphics import plot_labeled_spn
from algorithms.learning import learn_class_discriminative

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def get_traditional_spn(**kwargs):

    l_0 = Categorical(p=[0.5, 0.5], scope=0)
    l_1 = Categorical(p=[0.5, 0.5], scope=0)
    l_2 = Categorical(p=[0.5, 0.5], scope=1)
    l_3 = Categorical(p=[0.5, 0.5], scope=1)

    s_0 = Sum(weights=[0.5, 0.5], children=[l_0, l_1])
    s_1 = Sum(weights=[0.5, 0.5], children=[l_0, l_1])
    s_2 = Sum(weights=[0.5, 0.5], children=[l_2, l_3])
    s_3 = Sum(weights=[0.5, 0.5], children=[l_2, l_3])

    p_0 = Product(children=[s_0, s_2])
    p_1 = Product(children=[s_0, s_3])
    p_2 = Product(children=[s_1, s_3])

    spn = Sum(weights=[0.1, 0.3, 0.6], children=[p_0, p_1, p_2])

    assign_ids(spn)
    rebuild_scopes_bottom_up(spn)

    return spn

spn = get_traditional_spn()
assert is_valid(spn)

large_plot = True
margins = 0.1

fig = plt.figure(figsize=(5,5))

plot_labeled_spn(spn,
    f'src/report-vis/spn-structure-example.png',
    large=large_plot,
    save=False,
    margins=margins
)

fig.savefig(f'src/report-vis/spn-structure-example.png')
fig.savefig(f'src/report-vis/spn-structure-example.pdf')
