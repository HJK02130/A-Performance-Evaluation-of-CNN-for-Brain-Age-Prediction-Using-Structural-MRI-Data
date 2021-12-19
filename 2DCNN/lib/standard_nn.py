"""
Utility to create simple sequential networks for classification or regression

Create feed forward network with different `hidden_sizes`

Create convolution networks with different `channels` (hidden_size), (2,2) max pooling
"""
import typing

import numpy
import torch.nn as nn

from lib.utils.torch_utils import Reshape, infer_shape


# TODO : extend to cover pooling sizes, strides etc for conv nets
def get_arch(input_shape: typing.Union[numpy.array, typing.List], output_size: int,
             feed_forward: bool, hidden_sizes: typing.List[int],
             kernel_size: typing.Union[typing.List[int], int] = 3,
             non_linearity: typing.Union[typing.List[str], str, None] = "relu",
             norm: typing.Union[typing.List[str], str, None] = None,
             pooling: typing.Union[typing.List[str], str, None] = None) -> nn.Module:

    # general assertions
    n_layers = len(hidden_sizes)
    if n_layers > 0:
        if isinstance(non_linearity, list):
            assert len(non_linearity) == n_layers, "non linearity list is not same as hidden size"
            non_linearities = non_linearity
        else:
            non_linearities = [non_linearity] * n_layers

        if isinstance(norm, list):
            assert len(norm) == n_layers, "norm list is not same as hidden size"
            norms = norm
        else:
            norms = [norm] * n_layers
    else:
        norms = []
        non_linearities = []

    modules = []

    if feed_forward:
        modules.append(Reshape())
        insize = int(numpy.prod(input_shape))

        for nl, no, outsize in zip(non_linearities, norms, hidden_sizes):
            modules.append(nn.Linear(insize, outsize))

            if nl == "relu":
                modules.append(nn.ReLU())
            elif nl is None:
                pass
            else:
                raise Exception(f"non-linearity {nl} not implemented")

            if no == "bn":
                modules.append(nn.BatchNorm1d(outsize))
            elif no is None:
                pass
            else:
                raise Exception(f"norm {no} is not implemented")

            insize = outsize

        modules.append(nn.Linear(insize, output_size))
        return  {"net" : nn.Sequential(*modules)}

    # assertion specific to convolutions
    assert n_layers >= 1, "Number of layers has to be more than 1 for convolution"
    if isinstance(kernel_size, list):
        assert len(kernel_size) == n_layers, "kernel size is not same as hidden size"
        kernel_sizes = kernel_size
    else:
        kernel_sizes = [kernel_size] * n_layers

    if isinstance(pooling, list):
        assert len(pooling) == n_layers, "pooling size is not same as hidden size"
        poolings = pooling
    else:
        poolings = [pooling] * n_layers

    # convolutional layer with 3x3 convolutions
    inchannel = input_shape[0]
    for nl, no, outchannel, k, p in zip(non_linearities, norms, hidden_sizes, kernel_sizes,
                                        poolings):
        modules.append(nn.Conv2d(inchannel, outchannel, kernel_size=k))

        if nl == "relu":
            modules.append(nn.ReLU())
        elif nl is None:
            pass
        else:
            raise Exception(f"non-linearity {nl} is not implemented")

        if no == "bn":
            modules.append(nn.BatchNorm2d(outchannel))
        elif no is None:
            pass
        else:
            raise Exception(f"norm {no} is not implemented")

        if p == "max_pool":
            modules.append(nn.MaxPool2d(2))

        elif p is None:
            pass
        else:
            raise Exception(f"pooling {p} is not implemented")

        inchannel = outchannel

    output_shape = infer_shape(nn.Sequential(*modules).to("cpu"), input_shape)
    modules.append(Reshape())
    modules.append(nn.Linear(int(numpy.prod(output_shape)), output_size))
    return {"net" : nn.Sequential(*modules)}
