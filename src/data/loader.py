import numpy as np

import torchvision
import torch
import tqdm

import glob
import os


class DataSet(torch.utils.data.Dataset):

    def __init__(self, path, is_train=False, transform=None):
        self.is_train = is_train
        self.images = []
        self.labels = []

        self.transform = transform
        self._load_data(path)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        _image = self.images[idx, :, :, :]
        _label = self.labels[idx, :]

        if self.transform:
            _image = self.transform(_image)

        return _image, _label

    def _load_data(self, path):
        mode = 'train' if self.is_train else 'test'
        image_list = glob.glob(os.path.join(path, f'{mode}/*-image.npy'))
        label_list = glob.glob(os.path.join(path, f'{mode}/*-label.npy'))

        for image_path, label_path in tqdm.tqdm(zip(image_list, label_list), desc=f'load {mode} data set'):
            self.images.append(np.load(image_path))

            _label = np.load(label_path, allow_pickle=True)
            self.labels.append(np.array([(e['class_id'], e['azimuth_angle']) for e in _label]))

        self.images = np.vstack(self.images)
        self.labels = np.vstack(self.labels)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import torchvision.transforms as transforms
    import mstar

    data_root = 'D:\\ivs\\Project\\deep-learning\\mstar-test\\dataset'
    _data_set = DataSet(data_root, is_train=True, transform=transforms.ToTensor())

    data_loader = torch.utils.data.DataLoader(
        _data_set, batch_size=1, shuffle=True, num_workers=0
    )

    for i, batch in enumerate(data_loader):
        image, label = batch
        print(image.shape)

        fig = plt.figure(figsize=(8, 8))
        fig.suptitle(f'{mstar.classes_name[label[0, 0]]}, angle: {label[0, 1]}')
        fig.add_subplot(1, 1, 1)
        plt.imshow(image[0, 0, :, :])
        plt.show()
