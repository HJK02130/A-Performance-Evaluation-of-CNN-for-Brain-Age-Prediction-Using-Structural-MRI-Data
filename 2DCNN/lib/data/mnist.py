import torch
from torchvision.datasets import MNIST

from lib.data.util import DATA_FOLDER


class Mnist(MNIST):
    def __init__(self, root=f"{DATA_FOLDER}/mnist", train=True, transform=None,
                 target_transform=None, download=True, init_transform=None,
                 init_target_transform=None, seed=None, fraction=1.0):
        super().__init__(root, train=train, transform=transform, target_transform=target_transform,
                         download=download)

        if seed is not None:
            rng_state = torch.get_rng_state()
            torch.manual_seed(seed)

        N = len(self.data)
        n = None

        if 0 < fraction < 1.0:
            n = int(N * fraction)
        elif N > fraction > 1:
            n = int(fraction)
        if n:
            indices = torch.randperm(N)[:n]
            self.data, self.targets = self.data[indices], self.targets[indices]

        if init_transform:
            self.data = self.data = init_transform(self.data)
        if init_target_transform:
            self.targets = init_target_transform(self.targets)

        if seed is not None:
            torch.set_rng_state(rng_state)
