from absl import logging
from absl import flags
from absl import app

from multiprocessing import Pool
import numpy as np

import json
import glob
import os

import mstar

flags.DEFINE_string('image_root', default='dataset', help='')
flags.DEFINE_string('dataset', default='soc', help='')
flags.DEFINE_boolean('is_train', default=True, help='')
flags.DEFINE_boolean('use_phase', default=False, help='')
FLAGS = flags.FLAGS

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def generate(src_path, dst_path, is_train, use_phase):
    if not os.path.exists(dst_path):
        os.makedirs(dst_path, exist_ok=True)
    print(f'Target Name: {os.path.basename(dst_path)}')

    _mstar = mstar.MSTAR(is_train=is_train, use_phase=use_phase, patch_size=88, stride=1)

    image_list = glob.glob(os.path.join(src_path, '*'))

    for path in image_list:
        label, _images = _mstar.read(path)
        for i, _image in enumerate(_images):
            name = os.path.splitext(os.path.basename(path))[0]
            with open(os.path.join(dst_path, f'{name}-{i}.json'), mode='w', encoding='utf-8') as f:
                json.dump(label, f, ensure_ascii=False, indent=2)
            np.save(os.path.join(dst_path, f'{name}-{i}.npy'), _image)


def main(_):
    dataset_root = os.path.join(project_root, FLAGS.image_root, FLAGS.dataset)
    raw_root = os.path.join(dataset_root, 'raw')

    mode = 'train' if FLAGS.is_train else 'test'

    output_root = os.path.join(dataset_root, mode)
    if not os.path.exists(output_root):
        os.makedirs(output_root, exist_ok=True)

    arguments = [
        (
            os.path.join(raw_root, mode, target),
            os.path.join(output_root, target),
            FLAGS.is_train, FLAGS.use_phase
        ) for target in mstar.target_name
    ]

    with Pool(10) as p:
        p.starmap(generate, arguments)


if __name__ == '__main__':
    app.run(main)
