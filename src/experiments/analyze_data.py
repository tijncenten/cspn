from prep import get_data

import numpy as np

def run_test():
    datasets = ['redwine', 'whitewine', 'wine', 'bank', 'segment', 'german', 'vehicle', 'vowel', 'authent', 'diabetes', 'cmc', 'electricity', 'gesture', 'breast', 'krvskp', 'dna', 'robot', 'mice', 'dresses', 'texture', 'splice', 'jungle', 'phishing', 'fashion', 'mnist']

    data_analysis_folder = 'data'

    for dataset in datasets:
        np.random.seed(0)
        data, ncat = get_data(dataset)
        nr_data = len(data)
        nr_vars = len(ncat)
        np.random.shuffle(data)
        train_test_split = 0.7
        split = int(len(data) * train_test_split)
        train_data = data[:split]
        test_data = data[split:]
        nr_train = len(train_data)
        nr_test = len(test_data)

        p_types = ["Gaussian" if cat == 1 else "Categorical" for cat in ncat]

        print('test')

        with open(f'{data_analysis_folder}/{dataset}-stats.txt', mode='w') as f:
            f.write(f'Dataset statistics ({dataset})\n')
            f.write(f'{nr_data = }\n')
            f.write(f'{nr_vars = }\n')
            f.write(f'{nr_train = }\n')
            f.write(f'{nr_test = }\n')
            f.write(f'ncat = {",".join([str(int(cat)) for cat in ncat])}\n')
            f.write(f'p_types = {",".join(p_types)}\n')
