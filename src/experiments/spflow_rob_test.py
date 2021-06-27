from spn.algorithms.LearningWrappers import learn_parametric, learn_classifier
from spn.algorithms.MPE import mpe
# from spn.gpu.TensorFlow import optimize_tf
from spn.structure.leaves.parametric.Parametric import Categorical, Gaussian
from spn.structure.Base import Context, Sum, Product, Leaf
from spn.structure.Base import assign_ids, rebuild_scopes_bottom_up, get_number_of_nodes
from spn.io.Graphics import plot_spn, plot_spn2

from load_data import load_data
from prep import get_data
from experiments import spflow_rat_spn_test2
from experiments.settings import Settings
from algorithms.rob_inference import rob_log_likelihood
from algorithms.rob_conditional import rob_classification
from algorithms.learning import learn_class_discriminative
from algorithms.structure import is_spn_tree, compute_node_depth, compute_tree_nodes, get_number_of_parameters, get_structure_cycles, check_tractable_robustness
from algorithms.rat_spn import learn_rat_spn
from algorithms.graphics import plot_labeled_spn

from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import os
import csv
import time


def run_test(settings=None, spn=None):
    ###########
    ## SETUP ##
    ###########
    if settings is None:
        settings = Settings('cmc',
            build_rat_spn=False,
            class_discriminative=False,
            max_depth=None,
            filter_tree_nodes=False)


    dataset_name = settings.dataset_name
    results_folder = settings.results_folder
    filename_ext = settings.filename_ext

    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    ##########
    ## DATA ##
    ##########

    np.random.seed(0)
    data, ncat = get_data(dataset_name)
    print(data, ncat)

    # Unit vector normalization
    if settings.norm == 'unit':
        data[:,:-1] = data[:,:-1] / np.linalg.norm(data[:,:-1], axis=1, keepdims=True)
    elif settings.norm == 'zscore':
        data[:,:-1] = stats.zscore(data[:,:-1])
    elif settings.norm is not None:
        raise ValueError(f'normalization {settings.norm} not implemented')

    print(f'length: {len(data)}')
    np.random.shuffle(data)
    train_test_split = 0.7
    split = int(len(data) * train_test_split)
    print(split)

    train_data = data[:split]
    test_data = data[split:]
    test_y_data = test_data[:,-1].copy().astype(np.int64)
    test_data[:,-1] = np.nan
    p_types = [Gaussian if cat == 1 else Categorical for cat in ncat]

    #########
    ## SPN ##
    #########

    if spn is None:

        class_learner = learn_classifier
        if settings.class_discriminative:
            class_learner = learn_class_discriminative

        if not settings.build_rat_spn:
            spn = class_learner(train_data,
                Context(parametric_types=p_types).add_domains(train_data),
                learn_parametric, len(ncat) - 1, min_instances_slice=settings.min_instances_slice)
        else:
            # Alternative learner (RAT-SPN)
            spn = spflow_rat_spn_test2.run_test(settings=settings)

            # spn = class_learner(train_data,
            #     Context(parametric_types=p_types).add_domains(train_data),
            #     learn_rat_spn, len(ncat) - 1)
            # spn = optimize_tf(spn, train_data)

        # check_tractable_robustness(spn)
        # get_structure_cycles(spn)
        # return

        if settings.max_depth != None:
            compute_node_depth(spn)
        if settings.filter_tree_nodes:
            compute_tree_nodes(spn)
        # TODO: Also make filter for all nodes without class_var

    plt.figure(figsize=(20, 20))

    plot_labeled_spn(spn,
        f'{results_folder}/spn-l-{filename_ext}.png',
        label_depth=settings.max_depth,
        label_tree_node=settings.filter_tree_nodes
    )
    # TODO: Implement spn plotter with different colors for [node type, tree/DAG, rob_cond, etc.]

    # print('stop')
    # return

    # res = mpe(spn, test_data)
    # res = res[:,-1]
    # cor_count = 0
    # tot_count = 0
    # print(res, test_y_data)
    # for g, l in zip(res, test_y_data):
    #     if g == l:
    #         cor_count += 1
    #     tot_count += 1
    # print(cor_count, tot_count, cor_count / tot_count)

    # # Test CSPN min and max likelihood values from epsilon
    # from spn.algorithms.Inference import log_likelihood
    # ll = log_likelihood(spn, test_data[:10])
    # print('\nlog_likelihood:')
    # # print(ll)
    # np.set_printoptions(suppress=True)
    # print(np.exp(ll))
    # ll = rob_log_likelihood(spn, test_data[:10], eps=0.1)
    # print('\nrobust log_likelihood:')
    # # print(ll)
    # print(np.exp(ll))


    is_tree = is_spn_tree(spn)
    if is_tree:
        print(f'SPN is tree structure')
    else:
        print(f'SPN is non-tree structure')


    nr_nodes = get_number_of_nodes(spn)
    nr_sum = get_number_of_nodes(spn, Sum)
    nr_product = get_number_of_nodes(spn, Product)
    nr_leaf = get_number_of_nodes(spn, Leaf)
    nr_params = get_number_of_parameters(spn)

    with open(f'{results_folder}/spn-stats-{filename_ext}.csv', mode='w') as f:
        f.write(f'SPN statistics ({filename_ext})\n')
        f.write(f'nr_nodes: {nr_nodes}\n')
        f.write(f'nr_sum: {nr_sum}\n')
        f.write(f'nr_product: {nr_product}\n')
        f.write(f'nr_leaf: {nr_leaf}\n')
        f.write(f'nr_params: {nr_params}\n')

        if settings.rat_spn_config is not None:
            vals = settings.RatSpnConfig
            for name, val in vals:
                f.write(f'{name}: {val}\n')
        
        f.write(f'n_epochs: {settings.n_epochs}\n')
        f.write(f'batch_size: {settings.batch_size}\n')
        f.write(f'learning_rate: {settings.learning_rate}\n')


    # Test CSPN credal classification
    # samples = 100
    samples = len(test_data)

    res = mpe(spn, test_data[:samples])
    pred = res[:,-1].astype(np.int64)
    goal = test_y_data[:samples]

    rob_start_time = time.time()

    rob = rob_classification(
        spn,
        test_data[:samples],
        test_data.shape[1]-1,
        int(ncat[-1]),
        pred=pred,
        max_depth=settings.max_depth,
        filter_tree_nodes=settings.filter_tree_nodes,
        leaf_eps=settings.leaf_eps,
        progress=True,
    )

    rob_end_time = time.time()
    rob_time = rob_end_time - rob_start_time
    print(f'rob classification time: {rob_time}')
    nr_test = samples

    with open(f'{results_folder}/spn-time-{filename_ext}.csv', mode='w') as f:
        f.write(f'SPN timing ({filename_ext})\n')
        f.write(f'rob_time: {rob_time}\n')
        f.write(f'nr_test: {nr_test}\n')


    # Store the robustness values for the test data
    with open(f'{results_folder}/raw-data-{filename_ext}.csv', mode='w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(['rob', 'pred', 'goal', 'corr'])
        for r, p, g in zip(rob, pred, goal):
            csv_writer.writerow([r, p, g, 1 if p == g else 0])



    print(f'rob mean: {np.mean(rob)}')
    print(f'rob min: {np.min(rob)}')
    print(f'rob max: {np.max(rob)}')

    fig, ax = plt.subplots()
    ax.hist(rob)
    ax.grid()
    fig.savefig(f'{results_folder}/rob-hist-{filename_ext}.png')

    min_robs = np.arange(0, 0.51, 0.01)
    corr_count = []
    tot_count = []
    acc = []

    for min_rob in min_robs:
        rob_filter = rob >= min_rob
        corr_rob = pred[rob_filter] == goal[rob_filter]
        corr_count_rob = np.sum(corr_rob)
        corr_count.append(corr_count_rob)
        tot_count.append(len(corr_rob))
        acc.append(corr_count_rob / len(corr_rob))
        print(f'rob (>= {min_rob})', corr_count_rob, len(corr_rob), corr_count_rob / len(corr_rob))

    with open(f'{results_folder}/results-{filename_ext}.csv', mode='w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(['eps', 'corr', 'tot', 'acc'])
        for rob, corr, tot, a in zip(min_robs, corr_count, tot_count, acc):
            csv_writer.writerow([rob, corr, tot, a])


    fig, ax = plt.subplots()

    ax.plot(min_robs, acc)
    ax.grid()

    fig.savefig(f'{results_folder}/spflow-rob-test-{filename_ext}.png')

    return spn


if __name__ == '__main__':
    run_test()
