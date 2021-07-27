from skimage.util import shape

import numpy as np
import tqdm

import glob
import os

target_name = ('2S1', 'BMP2', 'BRDM2', 'BTR60', 'BTR70', 'D7', 'T62', 'T72', 'ZIL131', 'ZSU234')
serial_number = {
    'b01': 0,
    '9563': 1,
    'E-71': 2,
    'k10yt7532': 3,
    'c71': 4,
    '92v13015': 5,
    'A51': 6,

    '132': 7,
    'A64': 7,

    'E12': 8,
    'd08': 9
}


class MSTAR(object):

    def __init__(self, is_train=False, use_phase=False, chip_size=94, patch_size=88, stride=40):
        self.is_train = is_train
        self.use_phase = use_phase
        self.chip_size = chip_size
        self.patch_size = patch_size
        self.stride = stride

    def read(self, path):
        f = open(path, 'rb')
        _header = self._parse_header(f)
        _data = np.fromfile(f, dtype='>f4')
        f.close()

        h = eval(_header['NumberOfRows'])
        w = eval(_header['NumberOfColumns'])

        _data = _data.reshape(-1, h, w)
        _data = _data.transpose(1, 2, 0)
        _data = _data.astype(np.float32)
        if not self.use_phase:
            _data = np.expand_dims(_data[:, :, 0], axis=2)

        # _data = self._normalize(_data)
        _data = self._center_crop(_data)

        if self.is_train:
            _data = self._data_augmentation(_data, patch_size=self.patch_size, stride=self.stride)
        else:
            _data = [self._center_crop(_data, size=self.patch_size)]

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
        patches = shape.view_as_windows(_data, window_shape=(patch_size, patch_size, channels), step=stride)
        patches = patches.reshape(-1, patch_size, patch_size, channels)
        return patches

    @staticmethod
    def _extract_meta_label(header):

        target_type = header['TargetType']
        sn = header['TargetSerNum']

        class_id = serial_number[sn]

        azimuth_angle = MSTAR._get_azimuth_angle(header['TargetAz'])

        return {
            'class_id': class_id,
            'target_type': target_type,
            'serial_number': sn,
            'azimuth_angle': azimuth_angle
        }

    @staticmethod
    def _get_azimuth_angle(angle):
        azimuth_angle = eval(angle)
        if azimuth_angle > 180:
            azimuth_angle -= 180
        return int(azimuth_angle)

    @staticmethod
    def _normalize(x):
        d = (x - x.min()) / (x.max() - x.min())
        return d.astype(np.float32)
