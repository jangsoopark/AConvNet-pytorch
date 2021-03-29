from skimage.util.shape import view_as_windows
import numpy as np
import tqdm
import glob
import os

classes_name = ('2S1', 'BMP-2', 'BRDM-2', 'BTR-60', 'BTR-70', 'D7', 'T-62', 'T-72', 'ZIL-131', 'ZSU-234')
serial_numbers = ('b01', '9563', 'E-71', 'k10yt7532', 'c71', '92v13015', 'A51', '132', 'E12', 'd08')


class MSTAR(object):

    def __init__(self, is_train=False, patch_size=88, stride=40):
        self.is_train = is_train
        self.stride = stride
        self.patch_size = patch_size

    def read(self, path):
        f = open(path, 'rb')
        _header = self._parse_header(f)
        _data = np.fromfile(f, dtype='>f4')
        f.close()

        h = eval(_header['NumberOfRows'])
        w = eval(_header['NumberOfColumns'])

        _data = _data.reshape(-1, h, w)
        _data = _data.transpose(1, 2, 0)
        _data = self._center_crop(_data)
        if self.is_train:
            _data = self._data_augmentation(_data, patch_size=self.patch_size, stride=self.stride)
        else:
            _data = self._center_crop(_data, size=self.patch_size)

        meta_label = self._extract_meta_label(_header)
        return meta_label, _data

    @staticmethod
    def _parse_header(file):
        header = {}
        for line in file:
            line = line.decode('utf-8')
            line = line.strip()

            if not line:
                continue

            if 'PhoenixHeaderVer' in line:
                continue

            if 'EndofPhoenixHeader' in line:
                break

            key, value = line.split('=')
            header[key.strip()] = value.strip()

        return header

    @staticmethod
    def _center_crop(data, size=128):
        h, w, _ = data.shape

        y = (h - size) // 2
        x = (w - size) // 2

        return data[y: y + size, x: x + size]

    @staticmethod
    def _data_augmentation(data, patch_size=88, stride=40):
        # patch extraction
        _data = MSTAR._center_crop(data, size=94)
        _, _, channels = _data.shape
        patches = view_as_windows(_data, window_shape=(patch_size, patch_size, channels), step=stride)
        patches = patches.reshape(-1, patch_size, patch_size, channels)
        return patches

    @staticmethod
    def _extract_meta_label(header):

        target_type = header['TargetType']
        serial_number = header['TargetSerNum']

        class_id = serial_numbers.index(serial_number)

        azimuth_angle = MSTAR._get_azimuth_angle(header['TargetAz'])

        return {
            'class_id': class_id,
            'target_type': target_type,
            'serial_number': serial_number,
            'azimuth_angle': azimuth_angle
        }

    @staticmethod
    def _get_azimuth_angle(angle):
        azimuth_angle = eval(angle)
        if azimuth_angle > 180:
            azimuth_angle -= 180
        return int(azimuth_angle)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    data_root = 'D:\\ivs\\Project\\deep-learning\\mstar-test\\dataset'
    _mstar = MSTAR(is_train=True, stride=4)
    mode = 'train'
    for c in classes_name:
        data_list = glob.glob(os.path.join(data_root, f'raw/{mode}/{c}/*'))
        label, _image = _mstar.read(data_list[0])

        if _mstar.is_train:
            for i in range(_image.shape[0]):
                fig = plt.figure(figsize=(8, 4))
                fig.suptitle(f'{classes_name[label["class_id"]]}, angle: {label["azimuth_angle"]}')
                fig.add_subplot(1, 2, 1)
                plt.imshow(_image[i, :, :, 0], cmap='gray')
                fig.add_subplot(1, 2, 2)
                plt.imshow(_image[i, :, :, 1], cmap='gray')
                plt.show()
        else:
            fig = plt.figure(figsize=(8, 4))
            fig.add_subplot(1, 2, 1)
            plt.imshow(_image[:, :, 0], cmap='gray')
            fig.add_subplot(1, 2, 2)
            plt.imshow(_image[:, :, 1], cmap='gray')
            plt.show()
