import torch.nn as nn
import torch

from . import _blocks


class Network(nn.Module):

    def __init__(self, classes, dropout_rate=0.5):
        super(Network, self).__init__()
        self.dropout_rate = dropout_rate

        self._layer = nn.Sequential(
            _blocks.Conv2DBlock(shape=[5, 5, 2, 16], stride=1),
            _blocks.Conv2DBlock(shape=[5, 5, 16, 32], stride=1),
            _blocks.Conv2DBlock(shape=[6, 6, 32, 64], stride=1),

            nn.Conv2d(64, 128, kernel_size=(5, 5), stride=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.dropout_rate),

            nn.Conv2d(128, classes, kernel_size=(3, 3), stride=(1, 1)),
            nn.Flatten(),
        )

    def forward(self, x):
        return self._layer(x)
