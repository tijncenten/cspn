from experiments.settings import Settings

import numpy as np
import matplotlib.pyplot as plt
import os
import csv
import time

def run_test(settings_list=None):
    ###########
    ## SETUP ##
    ###########
    if settings_list is None:
        settings_list = []

        settings_list.append(
            Settings('cmc',
                build_rat_spn=False,
                class_discriminative=False)
        )

        settings_list.append(
            Settings('cmc',
                build_rat_spn=False,
                class_discriminative=True)
        )

    target_dir = 'results/plots'

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    ###########
    ## PLOTS ##
    ###########

    # Plot Accuracy and samples of the given model settings
    # plotted using results data
    paths = [f'{settings.results_folder}/results-{settings.filename_ext}.csv' for settings in settings_list]
    results = [read_file(path) for path in paths]

    fig, ax = plt.subplots()
    ax.grid()
    ax2 = ax.twinx()

    for res, settings in zip(results, settings_list):
        ax.plot(res[0], res[3], label=f'{settings.filename_ext}')
        ax2.plot(res[0], [s / res[2][0] for s in res[2]], '--')



    ax.set_ylim(bottom=0, top=1)
    ax.set_xlabel('$\epsilon$-robustness')
    ax.set_ylabel('Accuracy')
    

    ax2.set_ylim(bottom=0, top=1)
    ax2.set_ylabel('Samples (%)')

    
    ax.legend(loc='lower left')

    fig.savefig(f'{target_dir}/plot-{time.time()}.png')


    # Plot the accuracy of the mixed samples data (same set of samples for both results)
    # plotted using raw data
    paths = [f'{settings.results_folder}/raw-data-{settings.filename_ext}.csv' for settings in settings_list]
    raw_results = [read_raw_file(path) for path in paths]

    for sample_set_id in range(len(settings_list)):
        fig, ax = plt.subplots()
        ax.grid()

        rob = raw_results[sample_set_id][0]

        accs = [[] for i in range(len(settings_list))]
        eps = []

        for min_rob in results[0][0]:
            eps.append(min_rob)
            rob_filter = rob >= min_rob
            for i in range(len(settings_list)):
                pred = raw_results[i][1]
                goal = raw_results[i][2]
                corr_rob = pred[rob_filter] == goal[rob_filter]
                corr_count_rob = np.sum(corr_rob)
                accs[i].append(corr_count_rob / len(corr_rob))

        for i, (acc, settings) in enumerate(zip(accs, settings_list)):
            ax.plot(eps, acc, label=f'{i}: {settings.filename_ext}')



        ax.set_ylim(bottom=0, top=1)
        ax.set_xlabel(f'$\epsilon$-robustness (sample set {sample_set_id})')
        ax.set_ylabel('Accuracy')

        
        ax.legend(loc='lower left')

        fig.savefig(f'{target_dir}/plot-samples-{sample_set_id}-{time.time()}.png')

    print('done')



def read_file(path):
    min_robs = []
    corr_count = []
    tot_count = []
    acc = []

    with open(path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for i, row in enumerate(csv_reader):
            if i == 0:
                continue
            print(row)
            min_robs.append(float(row[0]))
            corr_count.append(int(row[1]))
            tot_count.append(int(row[2]))
            acc.append(float(row[3]))

    return (min_robs, corr_count, tot_count, acc)


def read_raw_file(path):
    rob = []
    pred = []
    goal = []
    corr = []

    with open(path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for i, row in enumerate(csv_reader):
            if i == 0:
                continue
            print(row)
            rob.append(float(row[0]))
            pred.append(int(row[1]))
            goal.append(int(row[2]))
            corr.append(int(row[3]))

    return (np.array(rob), np.array(pred).astype(np.int64), np.array(goal).astype(np.int64), np.array(corr))




if __name__ == '__main__':
    run_test()
