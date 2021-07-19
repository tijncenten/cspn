
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

# norm = None
norm = 'zscore'

# dataset = 'robot'
# dataset = 'diabetes'
# dataset = 'authent'
# dataset = 'gesture'
dataset = 'texture'
min_instances_slice = 300
# dataset = 'breast'

settings_list = []

labels = []

settings_list.append(Settings(dataset, min_instances_slice=min_instances_slice, norm=norm))
labels.append('Tree')

settings_list.append(Settings(dataset, min_instances_slice=min_instances_slice, norm=norm, class_discriminative=True))
labels.append('CD-Tree')

settings_list.append(
    Settings(dataset, class_discriminative=True,
        build_rat_spn=True, rat_spn_config=config, rat_spn_large=False,
        n_epochs=n_epochs, batch_size=batch_size, learning_rate=learning_rate, norm=norm)
)
labels.append('CD-DAG')


title = f'Comparison - {dataset}'
hide_title = True

store_as_final = True

plot_rob_results.run_test(settings_list, title, labels=labels, mixed_samples=True, file_prefix=dataset, store_as_final=store_as_final, hide_title=hide_title)
