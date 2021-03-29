import numpy as np

import glob
import tqdm
import os

import mstar


def train(path):
    _mstar = mstar.MSTAR(is_train=True, stride=1)
    patch_size = 88

    train_root = os.path.join(path, 'train')
    if not os.path.exists(train_root):
        os.makedirs(train_root)

    for c in mstar.classes_name:

        data_list = glob.glob(os.path.join(path, f'raw/train/{c}/*'))
        image_list = []
        label_list = []

        for data_path in tqdm.tqdm(data_list, desc=f'train: {c}'):
            if 'ERROR' in data_path:
                continue
            if data_path.endswith('.HTM'):
                continue

            label, image = _mstar.read(data_path)
            num_patches = image.shape[0]
            label_list += [label for _ in range(num_patches)]
            image_list.append(image)

        image_list = np.asarray(image_list).reshape(-1, patch_size, patch_size, 2)
        np.save(os.path.join(train_root, f'{c}-label.npy'), label_list)
        np.save(os.path.join(train_root, f'{c}-image.npy'), image_list)


def test(path):
    _mstar = mstar.MSTAR(is_train=False)
    patch_size = 88

    train_root = os.path.join(path, 'test')
    if not os.path.exists(train_root):
        os.makedirs(train_root)

    for c in mstar.classes_name:

        data_list = glob.glob(os.path.join(path, f'raw/test/{c}/*'))
        image_list = []
        label_list = []

        for data_path in tqdm.tqdm(data_list, desc=f'test: {c}'):
            if 'ERROR' in data_path:
                continue
            if data_path.endswith('.HTM'):
                continue

            label, image = _mstar.read(data_path)
            label_list.append(label)
            image_list.append(image)

        image_list = np.asarray(image_list).reshape(-1, patch_size, patch_size, 2)
        np.save(os.path.join(train_root, f'{c}-label.npy'), label_list)
        np.save(os.path.join(train_root, f'{c}-image.npy'), image_list)


if __name__ == '__main__':
    data_root = 'D:\\ivs\\Project\\deep-learning\\mstar-test\\dataset'
    train(data_root)
    test(data_root)
