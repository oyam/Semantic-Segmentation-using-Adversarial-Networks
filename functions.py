#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import numpy as np
import chainer.functions as F
from chainer import Variable
from chainer import cuda


def bilinear_interpolation_kernel(in_channels, out_channels, ksize):
    """calculate a bilinear interpolation kernel

    Args:
        in_channels (int): Number of channels of input arrays. If ``None``,
            parameter initialization will be deferred until the first forward
            data pass at which time the size will be determined.
        out_channels (int): Number of channels of output arrays.
        ksize (int): Size of filters (a.k.a. kernels).
    """

    factor = (ksize + 1) / 2
    if ksize % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:ksize, :ksize]
    k = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
                                                                                
    W = np.zeros((in_channels, out_channels, ksize, ksize)).astype(np.float32)
    W[range(in_channels), range(out_channels), :, :] = k
    return W


def crop_to_target(x, target):
    """Crop variable to target shape.

    Args:
        x (~chainer.Variable): Input variable of shape :math:`(n, c_I, h, w)`.
        target (~chainer.Variable): Variable with target output shape
            :math:`(n, h, w)` or `(n, c_I, h, w)`.
    """

    if target.ndim==3:
        t_h, t_w = target.shape[1], target.shape[2]
    elif target.ndim==4:
        t_h, t_w = target.shape[2], target.shape[3]
    cr = int((x.shape[2] - t_h) / 2)
    cc = int((x.shape[3] - t_w) / 2)
    x_cropped = x[:, :, cr:cr + t_h, cc:cc + t_w]
    return x_cropped
            

def global_average_pooling_2d(x, use_cudnn=True):
    """Spatial global average pooling function.

    Args:
        x (~chainer.Variable): Input variable.
        use_cudnn (bool): If ``True`` and cuDNN is enabled, then this function
            uses cuDNN as the core implementation.
    """

    return F.average_pooling_2d(x, ksize=(x.shape[2], x.shape[3]), use_cudnn=use_cudnn)
