
from spn.algorithms.Validity import is_valid
from spn.structure.leaves.parametric.Parametric import Categorical, Gaussian
from spn.structure.Base import Context, Sum, Product
from spn.structure.Base import assign_ids, rebuild_scopes_bottom_up

from algorithms.graphics import plot_labeled_spn
from algorithms.learning import learn_class_discriminative
from algorithms.nodes import CollapsedNode

import numpy as np
import matplotlib.pyplot as plt



def create_problem_structure(*args, **kwargs):
    # class_left = Categorical(p=[0.5, 0.5], scope=1)
    # spn_center = Sum(weights=[0.5, 0.5], children=[Gaussian(0, 1, scope=0), Gaussian(5, 1, scope=0)])
    # class_right = Categorical(p=[0.5, 0.5], scope=1)
    class_left = CollapsedNode(scope=1)
    spn_center = CollapsedNode(scope=0)
    class_right = CollapsedNode(scope=1)

    p_left = Product(children=[class_left, spn_center])
    p_right = Product(children=[spn_center, class_right])

    spn = Sum(weights=[0.5, 0.5], children=[p_left, p_right])

    return spn

def get_small_class_discriminative_structure(**kwargs):
    train_data = np.array([[0,0,0], [0,1,1], [1,0,0], [1,1,0]])
    p_types = [Categorical, Categorical, Categorical]

    sub1 = create_problem_structure()
    sub2 = create_problem_structure()
    c1 = Categorical(p=[0.5, 0.5], scope=2)
    c2 = Categorical(p=[0.5, 0.5], scope=2)


    p1 = Product(children=[sub1, c1])
    p2 = Product(children=[c2, sub2])
    spn = Sum(weights=[0.5, 0.5], children=[p1, p2])

    assign_ids(spn)
    rebuild_scopes_bottom_up(spn)

    return spn

spn = get_small_class_discriminative_structure()
assert is_valid(spn)

large_plot = True
scope_letters = [None, None, 'c']
margins = 0.2

fig = plt.figure()

plot_labeled_spn(spn,
    f'src/report-vis/cspn-cycle-example.png',
    large=large_plot,
    scope_letters=scope_letters,
    margins=margins,
    save=False
)

fig.text(0.25, 0.58, '$\mathcal{T}$', fontsize=22)
fig.text(0.24, 0.04, '$\mathcal{B}$', fontsize=22)
fig.text(0.36, 0.66, '$\mathcal{D}$', fontsize=22)

fig.text(0.09, 0.3, '$\mathcal{M}$', fontsize=22)
fig.text(0.37, 0.3, '$\mathcal{N}$', fontsize=22)


fig.savefig(f'src/report-vis/cspn-cycle-example.png')
fig.savefig(f'src/report-vis/cspn-cycle-example.pdf')


