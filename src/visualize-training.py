from absl import logging
from absl import flags
from absl import app

import matplotlib.pyplot as plt
import numpy as np
import json
import os

import utils

flags.DEFINE_string('experiments_path', os.path.join(utils.project_root, '../experiments'), help='')
flags.DEFINE_string('config_name', 'AConvNet.json', help='')
FLAGS = flags.FLAGS


def load_history(path):
    with open(path, mode='r', encoding='utf-8') as f:
        return json.load(f)


def main(_):
    experiments_path = FLAGS.experiments_path
    config_name = FLAGS.config_name

    # load configurations
    config = utils.load_config(os.path.join(experiments_path, config_name))
    model_name = config['model_name']

    h = load_history(os.path.join(experiments_path, f'history-{model_name}.json'))
    epoch = np.arange(len(h['loss']))

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    _p1, = ax1.plot(epoch, h['loss'], marker='.', c='blue', label='loss')
    _p2, = ax2.plot(epoch, h['accuracy'], marker='.', c='red', label='accuracy')
    plt.legend([_p1, _p2], ['loss', 'accuracy'], loc='upper right')

    plt.grid()

    ax1.set_xlabel('epoch')
    ax1.set_ylabel('loss', color='blue')
    ax2.set_ylabel('accuracy', color='red')
    plt.show()


if __name__ == '__main__':
    app.run(main)
