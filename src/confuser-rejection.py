from absl import logging
from absl import flags
from absl import app

from tqdm import tqdm

import matplotlib.pyplot as plt
import torchvision.transforms as transforms

import numpy as np
import torch
import json
import os

import data.loader as mstar
import model
import utils

flags.DEFINE_string('experiments_path', os.path.join(utils.project_root, '../experiments'), help='')
flags.DEFINE_string('data_path', os.path.join(utils.project_root, '../dataset'), help='')
flags.DEFINE_string('config_name', 'AConvNet-Confuser-Rejection.json', help='')
FLAGS = flags.FLAGS

logging.set_verbosity(logging.INFO)

utils.set_random_seed(777)


def load_data(path, is_train, batch_size):
    _data_set = mstar.ConfuserRejectionSet(path, is_train=is_train, transform=transforms.Compose([
        transforms.ToTensor()
    ]))
    return torch.utils.data.DataLoader(
        _data_set, batch_size=batch_size, shuffle=is_train, num_workers=0
    ), _data_set.num_known, _data_set.num_confuser


@torch.no_grad()
def confuser_rejection(num_classes, model_path):
    test_set, num_known, num_confuser = load_data(FLAGS.data_path, is_train=False, batch_size=1)
    m = model.Model(classes=num_classes)
    m.load(model_path)

    # Test loop

    p_d = []
    p_fa = []
    for th in np.arange(1, 0, -0.01):
        confuser = 0
        known = 0

        for i, data in enumerate(tqdm(test_set, desc=f'threshold: {th}')):
            images, labels = data

            predictions = m.inference(images)
            predictions[predictions < 0] = 0
            _softmax = torch.nn.Softmax(dim=1)
            p = _softmax(predictions)

            is_confuser = p < th
            if torch.sum(is_confuser) == 3:
                continue

            if labels[:, 0] < 3:
                known += 1
            else:
                confuser += 1

        p_d.append(known / num_known)
        p_fa.append(confuser / num_confuser)

    plt.plot(p_fa, p_d)
    plt.ylabel('probability of detection')
    plt.xlabel('probability of false alarm')
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.grid(True)
    plt.show()


def training(epochs, experiments_path, lr, m, model_name, training_set):
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

        # logging and save the model
        logging.info(f'Epoch: {epoch + 1:03d}/{epochs:03d} | loss={np.mean(_loss):.4f} | lr={lr} | ')

        hist['loss'].append(np.mean(_loss))

        if experiments_path:
            m.save(os.path.join(experiments_path, model_name, f'model-{epoch + 1:03d}.pth'))
    return hist


def run_training(epochs, batch_size, momentum, lr, lr_step, lr_decay, weight_decay, num_classes, model_name,
                 experiments_path=None):
    # Data set
    training_set = load_data(FLAGS.data_path, is_train=True, batch_size=batch_size)

    # Model
    m = model.Model(
        classes=num_classes, momentum=momentum, lr=lr, lr_step=lr_step, lr_decay=lr_decay, weight_decay=weight_decay
    )

    hist = training(epochs, experiments_path, lr, m, model_name, training_set)

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
        run_training(epochs, batch_size, momentum, lr, lr_step, lr_decay, weight_decay,
                     num_classes, model_name, experiments_path)

    confuser_rejection(num_classes, os.path.join(experiments_path, model_name, f'model-{epochs:03d}.pth'))


if __name__ == '__main__':
    app.run(main)
