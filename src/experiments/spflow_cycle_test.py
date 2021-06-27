from spn.algorithms.LearningWrappers import learn_parametric, learn_classifier
from spn.algorithms.Validity import is_valid
from spn.structure.leaves.parametric.Parametric import Categorical, Gaussian
from spn.structure.Base import Context, Sum, Product
from spn.structure.Base import assign_ids, rebuild_scopes_bottom_up
from spn.io.Graphics import plot_spn, plot_spn2

from algorithms.structure import is_spn_tree, get_structure_cycles, check_tractable_robustness
from algorithms.graphics import plot_labeled_spn
from algorithms.learning import learn_class_discriminative

import numpy as np
import matplotlib.pyplot as plt
import os


def run_test():
    # Create an example spn which does not work with conditional expectations

    # spn = get_small_class_discriminative_structure(alt_sum=False, alt_cycle=True)
    # spn = get_tree_structure(alt_sum=False, alt_cycle=True)
    # spn = get_small_dag_structure(alt_sum=False, alt_cycle=True)
    # spn = get_small_dag_structure_alt(alt_sum=False, alt_cycle=True)
    # spn = get_small_cycle_structure()
    spn = get_small_cycle_structure_alt()
    assert is_valid(spn)

    plot_labeled_spn(spn,
        f'spn-dag-example.png'
    )



    check_tractable_robustness(spn, class_var=1)
    # get_structure_cycles(spn)


def get_small_cycle_structure(**kwargs):
    # Returns an spn structure of the following shape
    #
    #             +
    #            / \            
    #           x   x            
    #          / \ / \           
    #         C   +   C          
    #         1  / \  1
    #           G   G
    #           0   0

    class_left = Categorical(p=[0.5, 0.5], scope=1)
    spn_center = Sum(weights=[0.5, 0.5], children=[Gaussian(0, 1, scope=0), Gaussian(5, 1, scope=0)])
    class_right = Categorical(p=[0.5, 0.5], scope=1)

    p_left = Product(children=[class_left, spn_center])
    p_right = Product(children=[spn_center, class_right])

    spn = Sum(weights=[0.5, 0.5], children=[p_left, p_right])


    assign_ids(spn)
    rebuild_scopes_bottom_up(spn)

    return spn

def get_small_cycle_structure_alt(**kwargs):
    # Returns an alternative spn structure of the following shape
    #
    #             +
    #            / \            
    #           x   x            
    #          / \ / \           
    #         C   +   C          
    #         1  / \  1
    #           G   G
    #           0   0

    class_left = Categorical(p=[0.5, 0.5], scope=1)
    spn_center = Sum(weights=[0.5, 0.5], children=[Gaussian(0, 1, scope=0), Gaussian(5, 1, scope=0)])
    class_right = Categorical(p=[0.5, 0.5], scope=1)

    other = Sum(weights=[0.5, 0.5], children=[class_left, class_right])

    spn = Product(children=[spn_center, other])


    assign_ids(spn)
    rebuild_scopes_bottom_up(spn)

    return spn


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


def get_tree_structure(**kwargs):
    # train_data = np.array([[0,0,0], [0,1,1], [1,0,2], [1,1,0]])
    train_data = np.c_[np.r_[np.random.normal(5, 1, (500, 2)), np.random.normal((13, 6), 1, (500, 2)),
                            np.random.normal(10, 1, (500, 2)), np.random.normal((4, 11), 1, (500, 2))],
                   np.r_[np.zeros((1000, 1)), np.ones((1000, 1))]]
    p_types = [Gaussian, Gaussian, Categorical]

    spn = learn_classifier(train_data,
        Context(parametric_types=p_types).add_domains(train_data),
        learn_parametric, 2)

    return spn


def get_small_class_discriminative_structure(**kwargs):
    def build_simple_spn(*args, **kwargs):
        # l1 = Product(children=[])
        # l2 = Product(children=[])
        # p = Sum(children=[l1, l2])
        p = Sum(children=[])
        return p
        # return get_small_cycle_structure()

    train_data = np.array([[0,0,0], [0,1,1], [1,0,2], [1,1,0]])
    p_types = [Categorical, Categorical, Categorical]

    spn = learn_class_discriminative(train_data,
        Context(parametric_types=p_types).add_domains(train_data),
        build_simple_spn, 2)

    return spn


def get_small_dag_structure(**kwargs):
    left1 = Categorical(p=[0.5, 0.5], scope=1)
    right1 = Categorical(p=[0.5, 0.5], scope=1)
    
    c0 = Gaussian(0, 1, scope=2)
    c1 = Gaussian(0, 1, scope=2)
    p0 = Product(children=[left1, c0])
    p1 = Product(children=[right1, c1])
    spn_center = Sum(weights=[0.5, 0.5], children=[p0, p1])

    left0 = Categorical(p=[0.5, 0.5], scope=0)

    right0 = Categorical(p=[0.5, 0.5], scope=0)

    p_left = Product(children=[left0, spn_center])
    p_right = Product(children=[spn_center, right0])

    spn = Sum(weights=[0.5, 0.5], children=[p_left, p_right])

    assign_ids(spn)
    rebuild_scopes_bottom_up(spn)

    return spn

def get_small_dag_structure_alt(**kwargs):
    left1 = Categorical(p=[0.5, 0.5], scope=1)
    right1 = Categorical(p=[0.5, 0.5], scope=1)
    
    c0 = Gaussian(0, 1, scope=2)
    c1 = Gaussian(0, 1, scope=2)
    p0 = Product(children=[left1, c0])
    p1 = Product(children=[right1, c1])
    spn_center = Sum(weights=[0.5, 0.5], children=[p0, p1])

    left0 = Categorical(p=[0.5, 0.5], scope=0)

    right0 = Categorical(p=[0.5, 0.5], scope=0)

    other = Sum(weights=[0.5, 0.5], children=[left0, right0])
    spn = Product(children=[spn_center, other])

    assign_ids(spn)
    rebuild_scopes_bottom_up(spn)

    return spn



def get_large_cycle_structure(alt_sum=False, alt_cycle=False, **kwargs):
    A_left = Gaussian(0, 1, scope=0)
    class_node = Categorical(p=[0.5, 0.5], scope=2)
    A_right = Gaussian(5, 1, scope=0)

    p_low_left = Product(children=[A_left, class_node])
    p_low_right = Product(children=[class_node, A_right])


    B_left = Gaussian(2, 1, scope=1)
    s_left = Sum(weights=[0.5, 0.5], children=[p_low_left, p_low_right])
    s_right = Sum(weights=[0.5, 0.5], children=[p_low_left, p_low_right])
    B_right = Gaussian(4, 1, scope=1)

    if alt_sum:
        # Alternative option: extra sum child
        alt_class_node = Categorical(p=[0.5, 0.5], scope=2)
        alt_A = Gaussian(3, 1, scope=0)
        alt_p = Product(children=[alt_A, alt_class_node])

        s_left.children.append(alt_p)

    p_upp_left = Product(children=[B_left, s_left])
    p_upp_right = Product(children=[s_right, B_right])

    spn = Sum(weights=[0.5, 0.5], children=[p_upp_left, p_upp_right])

    if alt_cycle:
        alt_B = Gaussian(3, 1, scope=1)
        alt_class_node = Categorical(p=[0.5, 0.5], scope=2)
        alt_p = Product(children=[alt_B, alt_class_node, A_left])

        spn.children.append(alt_p)
        spn.weights = [1./len(spn.children) for c in spn.children]

    assign_ids(spn)
    rebuild_scopes_bottom_up(spn)

    return spn
