""" Mathematical formulae for different expressions"""

import torch

from .torch_utils import EPSILON


def echo_mi(f, s):
    N = s.shape[0]
    s = s.view(N, -1)
    return -torch.log(torch.abs(s) + EPSILON).sum(dim=1)


def get_echo_clip_factor(num_samples):
    max_fx = 1
    d_max = num_samples

    return (2 ** (-23) / max_fx) ** (1.0 / d_max)
