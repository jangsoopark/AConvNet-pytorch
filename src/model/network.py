import torch.nn as nn
import torch

from . import _blocks


class Network(nn.Module):

    def __init__(self, classes, dropout_rate=0.5):
        super(Network, self).__init__()
        self.dropout_rate = dropout_rate

        self.init_w = nn.init.kaiming_uniform_
        self.init_b = nn.init.zeros_

        self._layer = nn.Sequential(
            _blocks.Conv2DBlock(shape=[5, 5, 1, 16], stride=1, initializer=(self.init_w, self.init_b)),
            _blocks.Conv2DBlock(shape=[5, 5, 16, 32], stride=1, initializer=(self.init_w, self.init_b)),
            _blocks.Conv2DBlock(shape=[6, 6, 32, 64], stride=1, initializer=(self.init_w, self.init_b)),

            nn.Conv2d(64, 128, kernel_size=(5, 5), stride=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.dropout_rate),

            nn.Conv2d(128, classes, kernel_size=(3, 3), stride=(1, 1)),
            nn.Flatten(),
        )

        self._initialize()

    def regularizer(self):
        # Only Feature Extraction layers are regularized
        _weight_norm = 0
        for i, e in enumerate(self._layer):
            if not hasattr(e, '_layer'):
                continue

            _weight_norm += e.regularizer()
        return _weight_norm

    def forward(self, x):
        return self._layer(x)

    def _initialize(self):

        for i, e in enumerate(self._layer):
            if hasattr(e, '_layer'):
                continue
            if hasattr(e, 'weight'):
                self.init_w(self._layer[i].weight, nonlinearity='relu')
            if hasattr(e, 'bias'):
                self.init_b(self._layer[i].bias)
