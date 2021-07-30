from skimage import transform
import numpy as np


class ToTensor(object):

    def __init__(self):
        pass

    def __call__(self, sample):
        _input = sample

        if len(_input.shape) < 3:
            _input = np.expand_dims(_input, axis=2)

        _input = _input.transpose((2, 0, 1))

        return _input


class RandomCrop(object):

    def __init__(self, size):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            assert len(size) == 2
            self.size = size

    def __call__(self, sample):
        _input = sample

        if len(_input.shape) < 3:
            _input = np.expand_dims(_input, axis=2)

        h, w, _ = _input.shape
        oh, ow = self.size

        dh = h - oh
        dw = w - ow
        y = np.random.randint(0, dh) if dh > 0 else 0
        x = np.random.randint(0, dw) if dw > 0 else 0
        oh = oh if dh > 0 else h
        ow = ow if dw > 0 else w

        return _input[y: y + oh, x: x + ow, :]


class CenterCrop(object):

    def __init__(self, size):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            assert len(size) == 2
            self.size = size

    def __call__(self, sample):
        _input = sample

        if len(_input.shape) < 3:
            _input = np.expand_dims(_input, axis=2)

        h, w, _ = _input.shape
        oh, ow = self.size
        y = (h - oh) // 2
        x = (w - ow) // 2

        return _input[y: y + oh, x: x + ow, :]
