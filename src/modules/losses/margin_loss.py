#!/usr/bin/env python
import torch
from allennlp.nn import Activation
from allennlp.common import FromParams


class MarginLoss(FromParams):
    def __init__(self,
                 margin: float = 1.0,
                 smoothing: bool = False,
                 transformation: Activation = None):
        super().__init__()
        self.margin = margin
        self.smoothing = smoothing
        self.activation = transformation

    def __call__(self, x1: torch.Tensor, x2: torch.Tensor):
        if self.smoothing:
            x1 = 2*x1/(x1 + x2)
            x2 = 2*x2/(x1 + x2)
        if self.activation:
            x1 = self.activation(x1)
            x2 = self.activation(x2)
        margin = (x1 - x2)
        return torch.max(torch.zeros_like(x1), self.margin - margin)
