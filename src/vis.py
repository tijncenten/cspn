
from experiments.settings import Settings
from experiments import plot_rob_results


from spn.experiments.RandomSPNs_layerwise.distributions import RatNormal
from spn.experiments.RandomSPNs_layerwise.rat_spn import RatSpnConfig

config = RatSpnConfig()
config.F = None # Number of input features/variables (filled in later)
config.R = 1 # Number of repetitions
config.D = 2 # The depth
config.I = 2 # Number of distributions for each scope at the leaf layer
config.S = 3 # Number of sum nodes at each layer
config.C = 1 # The number of classes
config.dropout = 0.0
config.leaf_base_class = RatNormal
config.leaf_base_kwargs = {}

n_epochs = 60
batch_size = 100
learning_rate = 1e-2
min_instances_slice = 200 # SPFlow default: 200

larger_interval = False
smaller_interval = True

assert not larger_interval or not smaller_interval


leaf_eps_list = [None, 2.0, 4.0, 6.0]
if larger_interval:
    leaf_eps_list = [None, 4.0, 8.0]
if smaller_interval:
    leaf_eps_list = [None, 0.1, 0.5, 1.0]

norm = 'zscore'

# dataset = 'robot'
# norm = None
# dataset = 'diabetes'
# dataset = 'authent'
dataset = 'gesture'
# dataset = 'texture'
# min_instances_slice = 300
# dataset = 'breast'



viss = []
viss.append('Tree')
viss.append('CD-Tree')
viss.append('CD-DAG')

for vis in viss:
    settings_list = []
    labels = []

    for leaf_eps in leaf_eps_list:
        if vis == 'Tree':
            settings_list.append(Settings(dataset, min_instances_slice=min_instances_slice, norm=norm, leaf_eps=leaf_eps))
        
        if vis == 'CD-Tree':
            settings_list.append(Settings(dataset, min_instances_slice=min_instances_slice, norm=norm, leaf_eps=leaf_eps, class_discriminative=True))
        
        if vis == 'CD-DAG':
            settings_list.append(
                Settings(dataset, class_discriminative=True,
                    build_rat_spn=True, rat_spn_config=config, rat_spn_large=False,
                    n_epochs=n_epochs, batch_size=batch_size, learning_rate=learning_rate, norm=norm, leaf_eps=leaf_eps)
            )

        labels.append(f'Stdev interval: {0.0 if leaf_eps is None else leaf_eps}')


    title = f'Soft evidence - {dataset} - {vis}'
    hide_title = True
    final_folder = f'Soft evidence - {dataset}'

    store_as_final = True
    large = True

    prefix = f'{dataset}-{vis}'
    if larger_interval:
        prefix += '-li'
    if smaller_interval:
        prefix += '-si'

    plot_rob_results.run_test(settings_list, title, labels=labels, mixed_samples=False, file_prefix=prefix, large=large, store_as_final=store_as_final, final_folder=final_folder, hide_title=hide_title)
