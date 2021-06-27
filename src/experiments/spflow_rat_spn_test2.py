import os
import random
import sys
import time

from scipy import stats
import imageio
import numpy as np
import skimage
import torch
import torchvision
from torch import nn
from torchvision import datasets, transforms

import spn.algorithms.Inference as inference
from spn.structure.Base import get_number_of_nodes
from spn.experiments.RandomSPNs_layerwise.distributions import RatNormal
from spn.experiments.RandomSPNs_layerwise.rat_spn import RatSpn, RatSpnConfig

from algorithms.structure import get_number_of_parameters
from algorithms.torch.class_discriminative_layer import ClassDiscriminativeLayer, CustomRatSpn
from algorithms.torch.layerwise_to_simple import layerwise_to_simple_spn
from prep import get_data
from experiments.settings import Settings

# src: https://github.com/SPFlow/SPFlow/blob/master/src/spn/experiments/RandomSPNs_layerwise/train_mnist.py

def run_test(settings=None, n_epochs=None, batch_size=None):
    ###########
    ## SETUP ##
    ###########
    if settings is None:
        settings = Settings('robot',
            build_rat_spn=False,
            class_discriminative=False,
            max_depth=None,
            filter_tree_nodes=False)

    dataset_name = settings.dataset_name
    results_folder = settings.results_folder
    filename_ext = settings.filename_ext

    if n_epochs is None:
        n_epochs = settings.n_epochs if settings.n_epochs is not None else 10
    
    if batch_size is None:
        batch_size = settings.batch_size if settings.batch_size is not None else 100

    ## TORCH ##

    from torch import optim, nn

    dev = "cpu"
    if dev == "cpu":
        device = torch.device("cpu")
        use_cuda = False
    else:
        device = torch.device("cuda:0")
        use_cuda = True
        torch.cuda.benchmark = True

    ##########
    ## DATA ##
    ##########

    np.random.seed(0)
    data, ncat = get_data(dataset_name)
    print(data, ncat)

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
    label_idx = len(ncat) - 1

    train_loader, test_loader = get_data_loaders(train_data, test_data, use_cuda, batch_size=batch_size, device=device)

    output_spns = []

    class_counts = []

    for label, count in zip(*np.unique(data[:, label_idx], return_counts=True)):
        class_counts.append(count)

        #########
        ## SPN ##
        #########

        # Setup RatSpnConfig
        config = settings.rat_spn_config
        if config is None:
            config = RatSpnConfig()
            config.F = len(ncat) - 1 # Number of input features/variables
            config.R = 1 # Number of repetitions
            config.D = 2 # The depth
            config.I = 2 # Number of distributions for each scope at the leaf layer
            config.S = 2 # Number of sum nodes at each layer
            config.C = 1 # The number of classes
            config.dropout = 0.0
            config.leaf_base_class = RatNormal
            config.leaf_base_kwargs = {}
            settings.rat_spn_config = config
        else:
            config.F = len(ncat) - 1

        # Construct RatSpn from config
        spn = CustomRatSpn(config)

        # spn = spn.to(device)
        # spn.train()

        output_spns.append(spn)

    w = [c / np.sum(class_counts) for c in class_counts]
    print(w)
    cd_spn = ClassDiscriminativeLayer(output_spns, weights=w)
    cd_spn = cd_spn.to(device)
    cd_spn.train()

    parameters = list(cd_spn.parameters())
    nr_params = count_params(cd_spn)
    for sub_spn in output_spns:
        parameters += list(sub_spn.parameters())
        nr_params += count_params(sub_spn)

    spn = cd_spn
    
    print("Using device:", device)
    print(cd_spn)
    print("Number of pytorch parameters: ", nr_params)

    # Define optimizer
    loss_fn = nn.CrossEntropyLoss()
    lr = settings.learning_rate if settings.learning_rate is not None else 1e-1
    optimizer = optim.Adam(parameters, lr=lr)

    log_interval = 100

    lmbda = 1.0

    for epoch in range(n_epochs):
        t_start = time.time()
        running_loss = 0.0
        running_loss_ce = 0.0
        running_loss_nll = 0.0

        # if epoch > 100:
        #     for g in optimizer.param_groups:
        #         g['lr'] = 1e-2

        for batch_index, (data, target) in enumerate(train_loader):
            # Send data to correct device
            data, target = data.float().to(device), target.long().to(device)
            data = data.view(data.shape[0], -1)

            # Reset gradients
            optimizer.zero_grad()

            # Inference
            output = cd_spn(data)

            # Compute loss
            # loss_ce = loss_fn(output, target)
            # loss_nll = -output.sum() / (data.shape[0] * len(ncat) - 1)

            # loss = (1 - lmbda) * loss_nll + lmbda * loss_ce

            # loss = -output.sum() / (data.shape[0] * len(ncat) - 1)
            loss = loss_fn(output, target)

            # Backprop
            loss.backward()
            optimizer.step()

            # Log stuff
            running_loss += loss.item()
            # running_loss_ce += loss_ce.item()
            # running_loss_nll += loss_nll.item()
            # if batch_index % log_interval == (log_interval - 1):
            #     pred = output.argmax(1).eq(target).sum().cpu().numpy() / data.shape[0] * 100
            #     print(
            #         "Train Epoch: {} [{: >5}/{: <5} ({:.0f}%)]\tLoss_ce: {:.6f}\tLoss_nll: {:.6f}\tAccuracy: {:.0f}%".format(
            #             epoch,
            #             batch_index * len(data),
            #             60000,
            #             100.0 * batch_index / len(train_loader),
            #             running_loss_ce / log_interval,
            #             running_loss_nll / log_interval,
            #             pred,
            #         ),
            #         end="\r",
            #     )
            #     running_loss = 0.0
            #     running_loss_ce = 0.0
            #     running_loss_nll = 0.0

        t_delta = time_delta_now(t_start)
        print("Train Epoch: {} took {} - loss: {}".format(epoch, t_delta, running_loss / count))
        if epoch % 5 == 4:
            print("Evaluating model ...")
            evaluate_model(cd_spn, device, train_loader, "Train", ncat)
            evaluate_model(cd_spn, device, test_loader, "Test", ncat)

    # Evaluation
    cd_spn.eval()

    # evaluate_model(cd_spn, device, train_loader, "Train", ncat)
    # evaluate_model(cd_spn, device, test_loader, "Test", ncat)


    # TEMP TESTING

    # original = cd_spn
    # cd_spn = cd_spn.class_spns[0]

    print('\n================\n')



    # simple_spn = layerwise_to_simple_spn(cd_spn, ncat, rat_spn=True, config=config)
    simple_spn = layerwise_to_simple_spn(cd_spn, ncat)
    print(simple_spn)
    nr_params = get_number_of_parameters(simple_spn)
    nr_nodes = get_number_of_nodes(simple_spn)
    print(f'nr_params: {nr_params}')
    print(f'nr_nodes: {nr_nodes}')

    validate_data = train_data[:20]
    validate_loader = torch.utils.data.DataLoader(
        CustomDataset(validate_data),
        batch_size=100,
        shuffle=False
    )
    validate_data = validate_data.copy()
    validate_y_data = validate_data[:,-1].copy().astype(np.int64)
    validate_data[:,-1] = np.nan

    num_data = len(validate_data)
    num_nodes = get_number_of_nodes(simple_spn)
    lls_matrix = np.zeros((num_data, num_nodes))

    child_ids = [c.id for c in simple_spn.children]
    child_logweights = np.log(simple_spn.weights)

    simple_result = inference.log_likelihood(simple_spn, validate_data, lls_matrix=lls_matrix)
    simple_output = lls_matrix[:,child_ids]
    simple_output_norm = simple_output[:] + child_logweights
    
    simple_output = simple_output_norm
    torch_output = None
    torch_ll = 0
    for data, target in validate_loader:
        data, target = data.float().to(device), target.long().to(device)
        data = data.view(data.shape[0], -1)

        # TEMP
        # x = cd_spn._leaf(data)
        # for layer in cd_spn._inner_layers:
        #     x = layer(x)
        # n, d, c, r = x.size()
        # assert d == 1  # number of features should be 1 at this point
        # x = x.view(n, d, c * r, 1)
        # x = cd_spn.root(x)
        # x = x.squeeze()
        output = cd_spn(data)

        if torch_output is None:
            torch_output = output.data.numpy()
        else:
            torch_output = np.concatenate((torch_output, output.data.numpy()))

    def softmax(x, axis=0):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return e_x / e_x.sum(axis=axis, keepdims=True)

    torch_output = torch_output.reshape(simple_output.shape)

    print(simple_output)
    print(torch_output)
    print('--------------')
    print(np.sum(simple_output - torch_output))
    print('--------------')
    
    simple_output = softmax(simple_output, axis=1)
    torch_output = softmax(torch_output, axis=1)

    relative_error = np.abs(simple_output / torch_output - 1)
    print(np.average(relative_error))

    simple_res = np.argmax(simple_output, axis=1)
    torch_res = np.argmax(torch_output, axis=1)

    print(np.sum(simple_res == torch_res))

    print("done")

    return simple_spn


def evaluate_model(model: torch.nn.Module, device, loader, tag, ncat) -> float:
    """
    Description for method evaluate_model.

    Args:
        model (nn.Module): PyTorch module.
        device: Execution device.
        loader: Data loader.
        tag (str): Tag for information.

    Returns:
        float: Tuple of loss and accuracy.
    """
    model.eval()
    loss_ce = 0
    loss_nll = 0
    correct = 0
    criterion = nn.CrossEntropyLoss(reduction="sum")
    with torch.no_grad():
        for data, target in loader:
            data, target = data.float().to(device), target.long().to(device)
            data = data.view(data.shape[0], -1)
            output = model(data)
            loss_ce += criterion(output, target).item()  # sum up batch loss
            loss_nll += -output.sum()
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()

    loss_ce /= len(loader.dataset)
    loss_nll /= len(loader.dataset) + len(ncat) - 1
    accuracy = 100.0 * correct / len(loader.dataset)

    print(
        "{} set: Average loss_ce: {:.4f} Average loss_nll: {:.4f}, Accuracy: {}/{} ({:.1f}%)".format(
            tag, loss_ce, loss_nll, correct, len(loader.dataset), accuracy
        )
    )


def count_params(model: torch.nn.Module) -> int:
    """
    Count the number of parameters in a model.

    Args:
        model (torch.nn.Module): PyTorch model.

    Returns:
        int: Number of learnable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx, :-1], self.data[idx, -1]


def get_data_loaders(train_data, test_data, use_cuda, device, batch_size):
    """
    Args:
        use_cuda: Use cuda flag.

    """
    kwargs = {"num_workers": 8, "pin_memory": True} if use_cuda else {}

    test_batch_size = batch_size

    # Train data loader
    train_loader = torch.utils.data.DataLoader(
        CustomDataset(train_data),
        batch_size=batch_size,
        shuffle=True,
        **kwargs,
    )

    # Test data loader
    test_loader = torch.utils.data.DataLoader(
        CustomDataset(test_data),
        batch_size=test_batch_size,
        shuffle=True,
        **kwargs,
    )
    return train_loader, test_loader


def set_seed(seed: int):
    """
    Set the seed globally for python, numpy and torch.

    Args:
        seed (int): Seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def time_delta_now(t_start: float) -> str:
    """
    Convert a timestamp into a human readable timestring.
    Args:
        t_start (float): Timestamp.

    Returns:
        Human readable timestring.
    """
    a = t_start
    b = time.time()  # current epoch time
    c = b - a  # seconds
    days = round(c // 86400)
    hours = round(c // 3600 % 24)
    minutes = round(c // 60 % 60)
    seconds = round(c % 60)
    millisecs = round(c % 1 * 1000)
    return f"{days} days, {hours} hours, {minutes} minutes, {seconds} seconds, {millisecs} milliseconds"

