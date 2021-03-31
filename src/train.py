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
flags.DEFINE_string('config_name', 'AConvNet-1.json', help='')
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
    m.net.eval()
    for i, data in enumerate(tqdm(test_set)):
        images, labels = data

        predictions = m.inference(images)

        _, predictions = torch.max(predictions.data, 1)
        labels = labels.type(torch.LongTensor)

        num_data += labels.size(0)
        corrects += (predictions == labels[:, 0].to(m.device)).sum().item()

    m.net.train()
    accuracy = 100 * corrects / num_data
    return accuracy


def run(epochs, batch_size, momentum, lr, lr_step, lr_decay, weight_decay, num_classes, model_name,
        experiments_path=None):
    # Data set
    training_set = load_data(FLAGS.data_path, is_train=True, batch_size=batch_size)
    test_set = load_data(FLAGS.data_path, is_train=False, batch_size=1)

    # Model
    m = model.Model(
        classes=num_classes, momentum=momentum, lr=lr, lr_step=lr_step, lr_decay=lr_decay, weight_decay=weight_decay
    )

    hist = {
        'loss': [],
        'accuracy': []
    }

    # Training loop
    for epoch in range(epochs):
        _loss = []

        # Train over data set
        for i, data in enumerate(tqdm(training_set)):
            images, labels = data
            labels = labels.type(torch.LongTensor)
            _loss.append(m.optimize(images, labels[:, 0]))

        # learning rate schedule
        if m.lr_scheduler:
            lr = m.lr_scheduler.get_last_lr()[0]
            m.lr_scheduler.step()

        accuracy = validation(m, test_set)
        # logging and save the model
        logging.info(f'Epoch: {epoch + 1:03d}/{epochs:03d} | loss={np.mean(_loss):.4f} | lr={lr} | accuracy={accuracy}')

        hist['loss'].append(np.mean(_loss))
        hist['accuracy'].append(accuracy)
        if experiments_path:
            m.save(os.path.join(experiments_path, model_name, f'model-{epoch + 1:03d}.pth'))

    with open(os.path.join(experiments_path, f'history-{model_name}.json'), mode='w', encoding='utf-8') as f:
        json.dump(hist, f, ensure_ascii=True, indent=2)


def main(_):
    # Flags
    experiments_path = FLAGS.experiments_path
    config_name = FLAGS.config_name

    # load configurations
    config = utils.load_config(os.path.join(experiments_path, config_name))
    model_name = config['model_name']
    num_classes = config['num_classes']
    batch_size = config['batch_size']
    epochs = config['epochs']

    lr = config['lr']
    lr_step = config['lr_step']
    lr_decay = config['lr_decay']

    weight_decay = config['weight_decay']

    momentum = config['momentum']

    # initialize experimental path
    if not os.path.exists(os.path.join(experiments_path, model_name)):
        os.makedirs(os.path.join(experiments_path, model_name), exist_ok=True)

    run(epochs, batch_size, momentum, lr, lr_step, lr_decay, weight_decay, num_classes, model_name, experiments_path)


if __name__ == '__main__':
    app.run(main)
