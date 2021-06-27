from typing import Optional
from spn.experiments.RandomSPNs_layerwise.rat_spn import RatSpnConfig

class Settings():
    
    def __init__(self,
        dataset=None,
        build_rat_spn=False,
        rat_spn_config : Optional[RatSpnConfig] = None,
        rat_spn_large=False,
        min_instances_slice=200,
        class_discriminative=False,
        max_depth=None,
        filter_tree_nodes=False,
        leaf_eps=None,
        n_epochs=10,
        batch_size=100,
        learning_rate=1e-1,
        norm=None,
    ):
        self.dataset_name = dataset
        self.build_rat_spn = build_rat_spn
        self.rat_spn_config = rat_spn_config
        self.rat_spn_large = rat_spn_large
        self.min_instances_slice = min_instances_slice
        self.class_discriminative = class_discriminative
        self.spn_type = 'DAG' if build_rat_spn else 'tree'
        self.max_depth = max_depth
        self.filter_tree_nodes = filter_tree_nodes
        self.leaf_eps = leaf_eps
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.norm = norm

    def _settings_ext(self, string):
        if self.min_instances_slice != 200:
            string += f'-{self.min_instances_slice}'
        if self.class_discriminative:
            string += '-cd'
        if self.rat_spn_large:
            string += '-large'
        if self.leaf_eps != None:
            string += f'-leaf_eps{self.leaf_eps}'
        if self.max_depth != None:
            string += f'-d{self.max_depth}'
        if self.filter_tree_nodes:
            string += '-ftn'
        if self.norm:
            string += f'-{self.norm}'
        return string

    @property
    def results_folder(self):
        folder = f'results/{self.dataset_name}/{self.spn_type}'
        folder = self._settings_ext(folder)
        return folder

    @property
    def filename_ext(self):
        filename = f'{self.dataset_name}-{self.spn_type}'
        filename = self._settings_ext(filename)
        return filename

    @property
    def RatSpnConfig(self):
        conf = self.rat_spn_config
        res = []
        res.append(('F', conf.F))
        res.append(('R', conf.R))
        res.append(('D', conf.D))
        res.append(('I', conf.I))
        res.append(('S', conf.S))
        res.append(('C', conf.C))
        return res
