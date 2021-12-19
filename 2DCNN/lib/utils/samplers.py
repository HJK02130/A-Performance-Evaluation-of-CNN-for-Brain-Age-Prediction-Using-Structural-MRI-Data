""" torch samplers for different distributions"""

import numpy as np
import torch
from scipy.linalg import circulant


def sample_gaussian(mean, sigma, tril_sigma=False):
    noise = torch.randn_like(mean)

    # we getting sigma
    if tril_sigma:
        z_sample = torch.bmm(sigma, noise.unsqueeze(dim=2)).squeeze() + mean
    else:
        z_sample = noise * sigma + mean
    return z_sample


def sample_echo(f, s, m=None, replace=False, pop=True):
    """
        f, s : are the outputs of encoder (shape : [B, Z] for f)
            s is shape [B, Z] or [B, Z, Z]
        tril_sigma: if we have s as diagonal matrix or lt matrix
        m : number of samples to consider to generate noise when replace
            is true (default to batch_size)
        replace : sampling with replacement or not (if sampling with
            replacement, pop is not considered)
        pop: If true, remove the sample to which noise is being added
        detach_noise_grad : detach gradient of noise or not
    """
    batch_size, z_size = f.shape[0], f.shape[1:]

    # get indices
    if not replace:
        indices = circulant(np.arange(batch_size))
        if pop:
            # just ignore the first column
            indices = indices[:, 1:]
        for i in indices:
            np.random.shuffle(i)
    else:
        m = batch_size if m is None else m
        indices = np.random.choice(batch_size, size=(batch_size, m), replace=True)

    f_arr = f[indices.reshape(-1)].view(indices.shape + z_size)
    s_arr = s[indices.reshape(-1)].view(indices.shape + z_size)

    epsilon = f_arr[:, 0] + torch.sum(f_arr[:, 1:] * torch.cumprod(s_arr[:, :-1], dim=1), dim=1)

    z_sample = f + s * epsilon
    return z_sample
