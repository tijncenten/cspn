import math
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

from spn.experiments.RandomSPNs_layerwise.rat_spn import RatSpn

class ClassDiscriminativeLayer(nn.Module):
    def __init__(self, class_spns, weights):
        super().__init__()

        self.class_spns = class_spns

        ws = torch.tensor(weights)
        self.weights = nn.Parameter(ws)

        self.out_shape = f"(N, 1)"


    def forward(self, x: torch.Tensor):
        
        lls = torch.zeros(x.shape[0], len(self.class_spns), device=x.device)
        logweights = F.log_softmax(self.weights, dim=0)
        for idx, class_spn in enumerate(self.class_spns):
            outputs = class_spn(x)
            outputs = outputs.squeeze()
            outputs = outputs + logweights[idx]
            lls[:, idx] = outputs
        return lls

class CustomRatSpn(RatSpn):
    def _randomize(self, x: torch.Tensor) -> torch.Tensor:
        return x
