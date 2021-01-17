#!/usr/bin/env python
import pytest
import torch
import numpy as np
from src.modules.distributions.divergence import generalized_js_divergence, generalized_j_divergence
from src.modules.distributions.divergence import squared_hellinger_distance, cosine_distance
from torch.distributions import Normal


@pytest.mark.parametrize('dim, mean, stddev', [(1, 5, 1), (1, 10, 1)])
def test_js_divergence(dim, mean, stddev):
    a = Normal(mean*torch.ones(dim), stddev*torch.ones(dim))
    b = Normal(-mean*torch.ones(dim), stddev*torch.ones(dim))
    assert generalized_js_divergence(a, b) > 0


@pytest.mark.parametrize('dim, mean, stddev', [(1, 5, 1), (1, 10, 1)])
def test_j_divergence(dim, mean, stddev):
    a = Normal(mean*torch.ones(dim), stddev*torch.ones(dim))
    b = Normal(-mean*torch.ones(dim), stddev*torch.ones(dim))
    assert generalized_j_divergence(a, b) > 0


@pytest.mark.parametrize('m1, m2, s1, s2', [(-1, -4, 2, 4)])
def test_multi(m1, m2, s1, s2):
    a = Normal(m1, s1)
    b = Normal(m2, s2)
    assert generalized_j_divergence(a, b) <= 1.0


@pytest.mark.parametrize('m1, m2, s1, s2, bdim, edim', [(-1, 1, 1, 1, 32, 64)])
def test_hellinger(m1, m2, s1, s2, bdim, edim):
    a = Normal(torch.ones(bdim, edim)*m1, torch.ones(bdim, edim)*s1)
    b = Normal(torch.ones(bdim, edim)*m2, torch.ones(bdim, edim)*s2)
    shdab = squared_hellinger_distance(a, b)
    assert shdab == 1 - np.exp(-0.125*edim*(m1-m2)**2)


@pytest.mark.parametrize('m1, m2, s1, s2, bdim, edim', [(-1, 1, 1, 1, 32, 64)])
def test_cosine(m1, m2, s1, s2, bdim, edim):
    a = Normal(torch.ones(bdim, edim)*m1, torch.ones(bdim, edim)*s1)
    b = Normal(torch.ones(bdim, edim)*m2, torch.ones(bdim, edim)*s2)
    cdist = cosine_distance(a, b)
    assert cdist >= 0.0 and cdist <= 1.0
