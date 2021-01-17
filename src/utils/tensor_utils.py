#!/usr/bin/env python

import torch
from torch.distributions import Normal


def batch_mask(obj: any, mask: torch.Tensor):
    """
    Given a complicated object that may be a tensor or
    that has a tensor
    mask them along the first dimension which is assumed to be
    the batch dimension
    """
    if isinstance(obj, torch.Tensor):
        return obj[mask]
    elif isinstance(obj, dict):
        return {key: batch_mask(value, mask) for key, value in obj.items()}
    elif isinstance(obj, Normal):
        return Normal(batch_mask(obj.mean, mask), batch_mask(obj.stddev, mask))
    else:
        return obj
