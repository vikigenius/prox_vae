#!/usr/bin/env python
import torch
from torch import nn, Size, distributions


class GaussianAdditiveNoise(nn.Module):
    def __init__(self, shape: Size, stddev: float = 1.0):
        super().__init__()
        self.noise = distributions.Normal(torch.zeros(shape), stddev*torch.ones(shape))

    def forward(self, x):
        return x + self.noise.expand(x.size()).sample().to(x.device)
