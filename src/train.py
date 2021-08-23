from absl import logging
from absl import flags
from absl import app

from tqdm import tqdm

from torch.utils import tensorboard

import torchvision
import torch

import numpy as np

import json
import os

from data import preprocess
from data import loader
from utils import common
import model

flags.DEFINE_string('experiments_path', os.path.join(common.project_root, 'experiments'), help='')
flags.DEFINE_string('config_name', 'config/AConvNet-SOC.json', help='')
FLAGS = flags.FLAGS


common.set_random_seed(12321)


def load_dataset(path, is_train, name, batch_size):
    transform = [preprocess.CenterCrop(88), torchvision.transforms.ToTensor()]
    if is_train:
        transform = [preprocess.RandomCrop(88), torchvision.transforms.ToTensor()]
    _dataset = loader.Dataset(
        path, name=name, is_train=is_train,
        transform=torchvision.transforms.Compose(transform)
    )
    data_loader = torch.utils.data.DataLoader(
        _dataset, batch_size=batch_size, shuffle=is_train, num_workers=1
    )
    return data_loader


@torch.no_grad()
def validation(m, ds):
    num_data = 0
    corrects = 0

    # Test loop
    m.net.eval()
    _softmax = torch.nn.Softmax(dim=1)
    for i, data in enumerate(tqdm(ds)):
        images, labels, _ = data

        predictions = m.inference(images)
        predictions = _softmax(predictions)

        _, predictions = torch.max(predictions.data, 1)
        labels = labels.type(torch.LongTensor)
        num_data += labels.size(0)
        corrects += (predictions == labels.to(m.device)).sum().item()

    accuracy = 100 * corrects / num_data
    return accuracy


def run(epochs, mode, dataset, classes, channels, batch_size,
        lr, lr_step, lr_decay, weight_decay, dropout_rate,
        model_name, experiments_path=None):
    train_set = load_dataset('dataset', True, dataset, batch_size)
    valid_set = load_dataset('dataset', False, dataset, batch_size)

    m = model.Model(
        mode=mode, classes=classes, dropout_rate=dropout_rate, channels=channels,
        lr=lr, lr_step=lr_step, lr_decay=lr_decay,
        weight_decay=weight_decay
    )

    model_path = os.path.join(experiments_path, f'model/{model_name}')
    if not os.path.exists(model_path):
        os.makedirs(model_path, exist_ok=True)

    history_path = os.path.join(experiments_path, 'history')
    if not os.path.exists(history_path):
        os.makedirs(history_path, exist_ok=True)

    history = {
        'loss': [],
        'accuracy': []
    }

    for epoch in range(epochs):
        _loss = []

        m.net.train()
        for i, data in enumerate(tqdm(train_set)):
            images, labels, _ = data
            _loss.append(m.optimize(images, labels))

        if m.lr_scheduler:
            lr = m.lr_scheduler.get_last_lr()[0]
            m.lr_scheduler.step()

        accuracy = validation(m, valid_set)

        logging.info(
            f'Epoch: {epoch + 1:03d}/{epochs:03d} | loss={np.mean(_loss):.4f} | lr={lr} | accuracy={accuracy:.2f}'
        )

        history['loss'].append(np.mean(_loss))
        history['accuracy'].append(accuracy)

        if experiments_path:
            m.save(os.path.join(model_path, f'model-{epoch + 1:03d}.pth'))

    with open(os.path.join(history_path, f'history-{model_name}.json'), mode='w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=True, indent=2)


def main(_):
    logging.info('Start')
    experiments_path = FLAGS.experiments_path
    config_name = FLAGS.config_name

    config = common.load_config(os.path.join(experiments_path, config_name))

    mode = config['mode']
    dataset = config['dataset']
    classes = config['num_classes']
    channels = config['channels']
    epochs = config['epochs']
    batch_size = config['batch_size']

    lr = config['lr']
    lr_step = config['lr_step']
    lr_decay = config['lr_decay']

    weight_decay = config['weight_decay']
    dropout_rate = config['dropout_rate']

    model_name = config['model_name']

    run(epochs, mode, dataset, classes, channels, batch_size,
        lr, lr_step, lr_decay, weight_decay, dropout_rate,
        model_name, experiments_path)

    logging.info('Finish')


if __name__ == '__main__':
    app.run(main)
