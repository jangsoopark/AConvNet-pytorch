import torch.nn as nn
import torch

import collections


class BaseBlock(nn.Module):
    def __init__(self, initializer=None):
        super(BaseBlock, self).__init__()
        self._layer: nn.Sequential

        if initializer:
            init_w, init_b = initializer
            self._initialize(init_w, init_b)

    def forward(self, x):
        return self._layer(x)

    def _initialize(self, init_w, init_b):

        for i, e in enumerate(self._layer):
            if e.__class__.__name__.startswith('BatchNorm'):
                continue

            if hasattr(e, 'weight'):
                init_w(self._layer[i].weight, nonlinearity='relu')

            if hasattr(e, 'bias'):
                init_b(self._layer[i].bias)


class Conv2DBlock(BaseBlock):

    def __init__(self, shape, stride, initializer=None):
        super(Conv2DBlock, self).__init__()

        h, w, in_channels, out_channels = shape
        self._layer = nn.Sequential(collections.OrderedDict(
            [
                ('conv', nn.Conv2d(in_channels, out_channels, kernel_size=(h, w), stride=stride)),
                ('relu', nn.ReLU(inplace=True)),
                ('max_pool', nn.MaxPool2d(kernel_size=2, stride=2))
            ]))

        if initializer:
            init_w, init_b = initializer
            self._initialize(init_w, init_b)

    def forward(self, x):
        return self._layer(x)
