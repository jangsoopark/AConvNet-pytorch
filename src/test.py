from absl import logging
from absl import flags
from absl import app

from tqdm import tqdm

import torchvision.transforms as transforms

import numpy as np
import torch
import json
import os

import data.preprocess as preprocess
import data.loader as mstar
import model
import utils

flags.DEFINE_string('experiments_path', os.path.join(utils.project_root, '../experiments'), help='')
flags.DEFINE_string('data_path', os.path.join(utils.project_root, '../dataset'), help='')
flags.DEFINE_string('config_name', 'AConvNet.json', help='')
flags.DEFINE_integer('epoch', 200, help='')
FLAGS = flags.FLAGS

logging.set_verbosity(logging.INFO)

utils.set_random_seed(777)


def load_data(path, is_train, batch_size):
    _data_set = mstar.DataSet(path, is_train=is_train, transform=transforms.Compose([
        transforms.ToTensor()
    ]))
    return torch.utils.data.DataLoader(
        _data_set, batch_size=batch_size, shuffle=is_train, num_workers=0
    )


@torch.no_grad()
def validation(m, test_set):
    num_data = 0
    corrects = 0
    # Test loop
    m.eval()
    for i, data in enumerate(tqdm(test_set)):
        images, labels = data

        predictions = m.inference(images)

        _, predictions = torch.max(predictions.data, 1)
        labels = labels.type(torch.LongTensor)

        num_data += labels.size(0)
        corrects += (predictions == labels[:, 0].to(m.device)).sum().item()

    m.train()
    accuracy = 100 * corrects / num_data
    return accuracy


@torch.no_grad()
def run(batch_size, num_classes, model_path=None):
    # Data set
    test_set = load_data(FLAGS.data_path, is_train=False, batch_size=batch_size)

    # Model
    m = model.Model(classes=num_classes)
    m.load(model_path)

    num_data = 0
    corrects = 0
    # Test loop
    m.net.eval()
    for i, data in enumerate(tqdm(test_set)):
        images, labels = data

        predictions = m.inference(images)

        _, predictions = torch.max(predictions.data, 1)
        labels = labels.type(torch.LongTensor)

        num_data += labels.size(0)
        corrects += (predictions == labels[:, 0].to(m.device)).sum().item()

    accuracy = 100 * corrects / num_data
    logging.info(f'Accuracy : {accuracy}%')


def main(_):
    # Flags
    experiments_path = FLAGS.experiments_path
    config_name = FLAGS.config_name
    epoch = FLAGS.epoch

    # load configurations
    config = utils.load_config(os.path.join(experiments_path, config_name))
    model_name = config['model_name']
    num_classes = config['num_classes']
    batch_size = config['batch_size']

    # initialize experimental path
    if not os.path.exists(os.path.join(experiments_path, model_name)):
        os.makedirs(os.path.join(experiments_path, model_name), exist_ok=True)

    run(batch_size, num_classes, os.path.join(experiments_path, model_name, f'model-{epoch:03d}.pth'))


if __name__ == '__main__':
    app.run(main)
