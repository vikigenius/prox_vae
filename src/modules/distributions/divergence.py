#!/usr/bin/env python
import torch
from allennlp.common import Registrable
from torch.distributions.kl import kl_divergence
from torch.distributions import Normal


class Divergence(Registrable):
    def __call__(self, d1: Normal, d2: Normal):
        raise NotImplementedError


@Divergence.register("generalized_js_divergence")
class GeneralizedJSDivergence(Divergence):
    def __call__(self, d1: Normal, d2: Normal):
        gs = 2*d1.stddev*d2.stddev/(d1.stddev + d2.stddev)
        gm = (0.5*d1.mean*(1/d1.stddev) + 0.5*d2.mean*(1/d2.stddev))*gs
        gn = Normal(gm, gs)
        d = 0.5*kl_divergence(d1, gn) + 0.5*kl_divergence(d2, gn)
        return d.sum(dim=-1).mean()


@Divergence.register("squared_hellinger_distance")
class SquaredHellingerDistance(Divergence):
    def __call__(self, d1: Normal, d2: Normal):
        """
        This function computes the squared hellinger distance between
        two Multivariate Gaussian Distributions.
        d1 and d2 both are assumed to have diagonal covariance matrices
        shape: (b, d)
        """
        m1 = d1.mean
        m2 = d2.mean
        s1 = d1.stddev
        s2 = d2.stddev
        fac1 = ((s1*s2).prod(dim=1))**0.25
        fac2 = ((0.5*(s1 + s2)).prod(dim=1))**0.5
        fac = fac1/fac2
        efac = torch.exp(-0.125*(((m1-m2)**2)*(2/(s1+s2))).sum(dim=1))
        return (1 - fac*efac).mean()


@Divergence.register("cosine_distance")
class CosineDistance(Divergence):
    def __init__(self,
                 use_mean_only: bool = False):
        super().__init__()
        self.use_mean_only = use_mean_only

    def __call__(self, d1: Normal, d2: Normal):
        """
        Computes cosine distance between means of two Normals
        """
        if self.use_mean_only:
            z1 = d1.mean
            z2 = d2.mean
        else:
            z1 = d1.rsample()
            z2 = d2.rsample()
        return 0.5 - 0.5*((z1*z2).sum(dim=1)/(z1.norm(dim=1)*z2.norm(dim=1))).mean()
