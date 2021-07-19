from experiments.settings import Settings

import numpy as np
import matplotlib.pyplot as plt
import os
import csv
import time

def run_test(settings_list=None, title=None, labels=None, mixed_samples=True, file_prefix=None, large=False, store_as_final=False, final_folder=None, hide_title=False):
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
    suffix = str(time.time())
    if store_as_final:
        final_folder = title.replace(' ', '') if final_folder is None else final_folder.replace(' ', '')
        target_dir = f'results/final/{final_folder}'
        suffix = 'final'

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

    for i, (res, settings) in enumerate(zip(results, settings_list)):
        label = f'{settings.filename_ext}'
        if labels is not None:
            label = labels[i]
        ax.plot(res[0], res[3], label=label)
        ax2.plot(res[0], [s / res[2][0] for s in res[2]], '--')


    # Add striped lines for samples to the legend
    # ax.plot([], [], '--', color='black', label='Samples (%)')

    if large:
        fontsize = 16
        title_fontsize = 22
        tick_fontsize = 12
        ax.tick_params(axis='x', labelsize=tick_fontsize)
        ax.tick_params(axis='y', labelsize=tick_fontsize)
        ax2.tick_params(axis='y', labelsize=tick_fontsize)
    else:
        fontsize = 12
        title_fontsize = 12

    margin = 0.05
    bottom = 0
    top = 1
    bottom = bottom - margin
    top = top + margin


    ax.set_ylim(bottom=bottom, top=top)
    ax.set_xlabel('$\epsilon$-robustness', fontsize=fontsize)
    ax.set_ylabel('Accuracy', fontsize=fontsize)

    if title is not None and not hide_title:
        ax.set_title(title, fontsize=title_fontsize)
    

    ax2.set_ylim(bottom=bottom, top=top)
    ax2.set_ylabel('Samples (frac.)', fontsize=fontsize)

    
    fig.legend(loc='lower left', bbox_to_anchor=(0,0), bbox_transform=ax.transAxes, fontsize=fontsize)

    prefix = '' if file_prefix is None else f'-{file_prefix}'
    fig.savefig(f'{target_dir}/plot{prefix}-{suffix}.png')
    fig.savefig(f'{target_dir}/plot{prefix}-{suffix}.pdf')


    if not mixed_samples:
        return


    # Plot the accuracy of the mixed samples data (same set of samples for both results)
    # plotted using raw data
    paths = [f'{settings.results_folder}/raw-data-{settings.filename_ext}.csv' for settings in settings_list]
    raw_results = [read_raw_file(path) for path in paths]

    figsize = (4,3)
    fontsize = 12
    title_fontsize = fontsize

    for sample_set_id in range(len(settings_list)):
        fig, ax = plt.subplots(figsize=figsize)
        fig.tight_layout(pad=2)
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
            label = f'{i}: {settings.filename_ext}'
            if labels is not None:
                label = labels[i]
            if i == sample_set_id:
                label += '*'
            ax.plot(eps, acc, label=label)



        ax.set_ylim(bottom=bottom, top=top)
        ax.set_xlabel(f'$\epsilon$-robustness', fontsize=fontsize)
        ax.set_ylabel('Accuracy', fontsize=fontsize)

        if title is not None and not hide_title:
            ax.set_title(f'{title} ({labels[sample_set_id]} samples)', fontsize=title_fontsize)

        
        ax.legend(loc='lower left', fontsize=fontsize)

        fig.savefig(f'{target_dir}/plot-samples{prefix}-{sample_set_id}-{suffix}.png')
        fig.savefig(f'{target_dir}/plot-samples{prefix}-{sample_set_id}-{suffix}.pdf')

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
