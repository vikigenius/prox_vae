#!/usr/bin/env python
import torch
from src.modules.ops import GaussianAdditiveNoise


def test_gaussian_noise():
    noiselayer = GaussianAdditiveNoise((1, 128), 0.3)
    x = torch.zeros(32, 128)
    y = noiselayer(x)
    assert x.size() == y.size()
