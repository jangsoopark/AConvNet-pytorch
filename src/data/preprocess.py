import torchvision.transforms as transforms
import numpy as np


class SAMPLE(object):

    def __init__(self):
        pass

    def __call__(self, samples):
        image, label = samples

        return image, label
