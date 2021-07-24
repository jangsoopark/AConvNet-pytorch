import numpy as np

import torchvision
import torch
import tqdm

import json
import glob
import os

# import utils.common as common
project_root = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


class Dataset(torch.utils.data.Dataset):

    def __init__(self, path, name='soc', is_train=False, transform=None):
        self.is_train = is_train
        self.name = name

        self.images = []
        self.labels = []

        self.transform = transform
        self._load_data(path)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        _image = self.images[idx]
        _label = self.labels[idx]

        if self.transform:
            _image = self.transform(_image)

        return _image, _label

    def _load_data(self, path):
        mode = 'train' if self.is_train else 'test'

        image_list = glob.glob(os.path.join(project_root, path, f'{self.name}/{mode}/*/*.npy'))
        label_list = glob.glob(os.path.join(project_root, path, f'{self.name}/{mode}/*/*.json'))
        image_list = sorted(image_list, key=os.path.basename)
        label_list = sorted(label_list, key=os.path.basename)

        for image_path, label_path in tqdm.tqdm(zip(image_list, label_list), desc=f'load {mode} data set'):
            self.images.append(np.load(image_path))

            with open(label_path, mode='r', encoding='utf-8') as f:
                _label = json.load(f)

            self.labels.append(_label['class_id'])
