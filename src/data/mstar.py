from skimage.util import shape

import numpy as np

target_name_soc = ('2S1', 'BMP2', 'BRDM2', 'BTR60', 'BTR70', 'D7', 'T62', 'T72', 'ZIL131', 'ZSU234')
target_name_eoc_1 = ('2S1', 'BRDM2', 'T72', 'ZSU234')

target_name_eoc_2 = ('BMP2', 'BRDM2', 'BTR70', 'T72')
target_name_eoc_2_cv = ('T72-A32', 'T72-A62', 'T72-A63', 'T72-A64', 'T72-S7')
target_name_eoc_2_vv = ('BMP2-9566', 'BMP2-C21', 'T72-812', 'T72-A04', 'T72-A05', 'T72-A07', 'T72-A10')

target_name_confuser_rejection = ('BMP2', 'BTR70', 'T72', '2S1', 'ZIL131')

target_name = {
    'soc': target_name_soc,
    'eoc-1': target_name_eoc_1,
    'eoc-1-t72-132': target_name_eoc_1,
    'eoc-1-t72-a64': target_name_eoc_1,
    'eoc-2-cv': target_name_eoc_2 + target_name_eoc_2_cv,
    'eoc-2-vv': target_name_eoc_2 + target_name_eoc_2_vv,
    'confuser-rejection': target_name_confuser_rejection
}

serial_number = {
    'b01': 0,

    '9563': 1,
    '9566': 1,
    'c21': 1,

    'E-71': 2,
    'k10yt7532': 3,
    'c71': 4,
    '92v13015': 5,
    'A51': 6,

    '132': 7,
    '812': 7,
    's7': 7,
    'A04': 7,
    'A05': 7,
    'A07': 7,
    'A10': 7,
    'A32': 7,
    'A62': 7,
    'A63': 7,
    'A64': 7,

    'E12': 8,
    'd08': 9
}


class MSTAR(object):

    def __init__(self, name='soc', is_train=False, use_phase=False, chip_size=94, patch_size=88, stride=40):
        self.name = name
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

    def _data_augmentation(self, data, patch_size=88, stride=40):
        # patch extraction
        _data = MSTAR._center_crop(data, size=self.chip_size)
        _, _, channels = _data.shape
        patches = shape.view_as_windows(_data, window_shape=(patch_size, patch_size, channels), step=stride)
        patches = patches.reshape(-1, patch_size, patch_size, channels)
        return patches

    def _extract_meta_label(self, header):

        target_type = header['TargetType']
        sn = header['TargetSerNum']

        class_id = serial_number[sn]
        if not self.name == 'soc':
            class_id = target_name[self.name].index(target_name_soc[class_id])

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
