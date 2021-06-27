import os
import random
import sys
import time

import imageio
import numpy as np
import skimage
import torch
import torchvision
from torch import nn
from torchvision import datasets, transforms

from spn.experiments.RandomSPNs_layerwise.distributions import RatNormal
from spn.experiments.RandomSPNs_layerwise.rat_spn import RatSpn, RatSpnConfig

from algorithms.torch.class_discriminative_layer import ClassDiscriminativeLayer
from prep import get_data
from experiments.settings import Settings

# src: https://github.com/SPFlow/SPFlow/blob/master/src/spn/experiments/RandomSPNs_layerwise/train_mnist.py

def run_test(settings=None, n_epochs=None, batch_size=None):
    ###########
    ## SETUP ##
    ###########
    if settings is None:
        settings = Settings('dna',
            build_rat_spn=False,
            class_discriminative=False,
            max_depth=None,
            filter_tree_nodes=False)

    dataset_name = settings.dataset_name
    results_folder = settings.results_folder
    filename_ext = settings.filename_ext

    if n_epochs is None:
        n_epochs = 100
    
    if batch_size is None:
        batch_size = 100

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

    print(f'length: {len(data)}')
    np.random.shuffle(data)
    train_test_split = 0.7
    split = int(len(data) * train_test_split)
    print(split)

    train_data = data[:split]
    test_data = data[split:]
    label_idx = len(ncat) - 1

    output_spns = []

    class_counts = []

    for label, count in zip(*np.unique(data[:, label_idx], return_counts=True)):
        class_counts.append(count)
        # Filter out the relevant train and test data
        train = train_data[train_data[:, label_idx] == label, :]
        test = test_data[test_data[:, label_idx] == label, :]
        train_loader, test_loader = get_data_loaders(train, test, use_cuda, batch_size=batch_size, device=device)
        

        #########
        ## SPN ##
        #########

        # Setup RatSpnConfig
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

        # Construct RatSpn from config
        spn = RatSpn(config)

        spn = spn.to(device)
        spn.train()

        print("Using device:", device)

        print(spn)
        print("Number of pytorch parameters: ", count_params(spn))

        # Define optimizer
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.Adam(spn.parameters(), lr=1e-1)

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
                output = spn(data)

                # Compute loss
                # loss_ce = loss_fn(output, target)
                # loss_nll = -output.sum() / (data.shape[0] * len(ncat) - 1)

                # loss = (1 - lmbda) * loss_nll + lmbda * loss_ce

                loss = -output.sum() / (data.shape[0] * len(ncat) - 1)
                # loss = loss_fn(output, target)

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
            # if epoch % 5 == 4:
            #     print("Evaluating model ...")
            #     evaluate_model(spn, device, train_loader, "Train", ncat)
            #     evaluate_model(spn, device, test_loader, "Test", ncat)

        output_spns.append(spn)

    print(len(output_spns))

    # Evaluation

    train_loader, test_loader = get_data_loaders(train_data, test_data, use_cuda, batch_size=batch_size, device=device)

    w = [c / np.sum(class_counts) for c in class_counts]
    print(w)
    cd_spn = ClassDiscriminativeLayer(output_spns, weights=w)
    spn = spn.to(device)
    spn.eval()

    evaluate_model(cd_spn, device, train_loader, "Train", ncat)
    evaluate_model(cd_spn, device, test_loader, "Test", ncat)


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

