""" general utility function for data: mostly transformations """

import logging
import os
import random

import numpy
import numpy as np
from PIL import ImageFilter

logger = logging.getLogger()
DATA_FOLDER = os.getenv("DATA") if os.getenv("DATA") else "data"


def uniform_label_noise(p, labels, seed=None):
    if seed is not None:
        rng_state = numpy.random.get_state()
        numpy.random.seed(seed)

    labels = numpy.array(labels.tolist())
    N = len(labels)
    lst = numpy.unique(labels)

    # generate random labels
    rnd_labels = numpy.random.choice(lst, size=N)

    flip = numpy.random.rand(N) <= p
    labels = labels * (1 - flip) + rnd_labels * flip

    if seed is not None:
        numpy.random.set_state(rng_state)

    return labels


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=None):
        if sigma is None:
            sigma = [0.1, 2.0]
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


def lines_to_np_array(lines):
    return np.array([[int(i) for i in line.split()] for line in lines])


def load_binary_mnist():
    with open(os.path.join(DATA_FOLDER, "binary-mnist", "binarized_mnist_train.amat")) as f:
        lines = f.readlines()
    train_data = lines_to_np_array(lines).astype("float32")
    with open(os.path.join(DATA_FOLDER, "binary-mnist", "binarized_mnist_valid.amat")) as f:
        lines = f.readlines()
    validation_data = lines_to_np_array(lines).astype("float32")
    with open(os.path.join(DATA_FOLDER, "binary-mnist", "binarized_mnist_test.amat")) as f:
        lines = f.readlines()
    test_data = lines_to_np_array(lines).astype("float32")

    return {"train": train_data, "valid": validation_data, "test": test_data}


def load_mnist():
    import gzip
    import _pickle

    train, valid, test = _pickle.load(
        gzip.open(os.path.join(DATA_FOLDER, "mnist", "mnist.pkl.gz")), encoding="latin1",
    )
    return {"train": train, "valid": valid, "test": test}
