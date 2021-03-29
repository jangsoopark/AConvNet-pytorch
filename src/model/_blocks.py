import torch.nn as nn
import torch

import collections


class BaseBlock(nn.Module):
    def __init__(self):
        super(BaseBlock, self).__init__()
        self._layer: nn.Sequential

    def forward(self, x):
        return self._layer(x)


class Conv2DBlock(BaseBlock):

    def __init__(self, shape, stride):
        super(Conv2DBlock, self).__init__()

        h, w, in_channels, out_channels = shape
        self._layer = nn.Sequential(collections.OrderedDict(
            [
                ('conv', nn.Conv2d(in_channels, out_channels, kernel_size=(h, w), stride=stride)),
                ('relu', nn.ReLU(inplace=True)),
                ('max_pool', nn.MaxPool2d(kernel_size=2, stride=2))
            ]))

    def forward(self, x):
        return self._layer(x)
