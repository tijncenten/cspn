from spn.algorithms.Validity import is_valid
from spn.structure.leaves.parametric.Parametric import Categorical, Gaussian
from spn.structure.Base import Context, Sum, Product
from spn.structure.Base import assign_ids, rebuild_scopes_bottom_up

from algorithms.graphics import plot_labeled_spn
from algorithms.learning import learn_class_discriminative

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

annotated = True

def create_structure():
    class_left = Categorical(p=[0.5, 0.5], scope=1)
    spn_center = Sum(weights=[0.5, 0.5], children=[Gaussian(0, 1, scope=0), Gaussian(5, 1, scope=0)])
    class_right = Categorical(p=[0.5, 0.5], scope=1)

    p_left = Product(children=[class_left, spn_center])
    p_right = Product(children=[spn_center, class_right])

    spn = Sum(weights=[0.5, 0.5], children=[p_left, p_right])

    assign_ids(spn)
    rebuild_scopes_bottom_up(spn)

    class_left._superscript = 'a'
    class_right._superscript = 'b'
    spn_center.children[0]._superscript = 'c'
    spn_center.children[1]._superscript = 'd'

    return spn


spn = create_structure()
assert is_valid(spn)

large_plot = False
margins = 0.3

fig = plt.figure()

plot_labeled_spn(spn,
    f'src/report-vis/spn-problem-example.png',
    large=large_plot,
    save=False,
    margins=margins
)

fig.text(0.11, 0.3, '$f(X_1^a) = -1$', fontsize=12)
fig.text(0.76, 0.3, '$f(X_1^b) = 1$', fontsize=12)

fig.text(0.25, 0.1, '$I(X_0^c)$', fontsize=12)
fig.text(0.68, 0.1, '$I(X_0^d)$', fontsize=12)

fig.text(0.4, 0.7, '$w_1$')
fig.text(0.565, 0.7, '$w_2$')
fig.text(0.36, 0.3, '$w_3$')
fig.text(0.61, 0.3, '$w_4$')

if annotated:
    fig.text(0.48, 0.32, '$\mathcal{B}$', fontsize=16)
    fig.text(0.34, 0.6, '$\mathcal{L}$', fontsize=16)
    fig.text(0.63, 0.6, '$\mathcal{R}$', fontsize=16)
    fig.text(0.49, 0.85, '$\mathcal{T}$', fontsize=16)
    fig.savefig(f'src/report-vis/spn-problem-example-annotated.png')
    fig.savefig(f'src/report-vis/spn-problem-example-annotated.pdf')
    exit()


fig.savefig(f'src/report-vis/spn-problem-example.png')
fig.savefig(f'src/report-vis/spn-problem-example.pdf')
