from spn.algorithms.Validity import is_valid
from spn.structure.Base import Sum, Product, assign_ids, rebuild_scopes_bottom_up
from spn.structure.leaves.parametric.Parametric import Categorical

import numpy as np

def learn_class_discriminative(data, ds_context, spn_learn_wrapper, label_idx, **kwargs):
    spn = Sum()
    for label, count in zip(*np.unique(data[:, label_idx], return_counts=True)):
        prod = Product()
        k = int(np.max(ds_context.domains[label_idx]) + 1)
        p = np.zeros(k)
        p[int(label)] += 1
        class_label_node = Categorical(p=p, scope=label_idx)
        prod.children.append(class_label_node)

        # First filter to only include correct label data
        branch_data = data[data[:, label_idx] == label, :]
        # Then delete the column of label values
        branch_data = np.delete(branch_data, label_idx, axis=1)


        branch = spn_learn_wrapper(branch_data, ds_context, **kwargs)
        prod.children.append(branch)

        spn.children.append(prod)
        spn.weights.append(count / data.shape[0])

    # spn.scope.extend(branch.scope)
    assign_ids(spn)
    rebuild_scopes_bottom_up(spn)

    assert is_valid(spn)
    return spn

