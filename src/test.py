from absl import logging
from absl import flags
from absl import app
from tqdm import tqdm

import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torch

import numpy as np
import os

import data.loader as data_loader
import data.mstar as mstar
import model
import utils

flags.DEFINE_string('experiments_path', os.path.join(utils.project_root, '../experiments'), help='')
flags.DEFINE_string('data_path', os.path.join(utils.project_root, '../dataset'), help='')
flags.DEFINE_string('config_name', 'AConvNet.json', help='')
flags.DEFINE_integer('epoch', 42, help='')
FLAGS = flags.FLAGS

logging.set_verbosity(logging.INFO)

utils.set_random_seed(777)


def load_data(path, is_train, batch_size):
    _data_set = data_loader.DataSet(path, is_train=is_train, transform=transforms.Compose([
        transforms.ToTensor()
    ]))
    return torch.utils.data.DataLoader(
        _data_set, batch_size=batch_size, shuffle=is_train, num_workers=0
    )


def write_evaluation_log(path, confusion_matrix):
    dim = confusion_matrix.shape[0]
    accuracy = np.sum(confusion_matrix * np.eye(dim)) / np.sum(confusion_matrix)
    with open(path, mode='w', encoding='utf-8') as f:
        f.write(f'Accuracy: {accuracy}\n')
        f.write(f'class, {", ".join(mstar.classes_name)}\n')

        for i, name in enumerate(mstar.classes_name):
            f.write(f'{name}, {",".join(str(int(v)) for v in confusion_matrix[i, :])}\n')


def write_fp_image(path, label, predict, index, image):
    _label_name = mstar.classes_name[label]
    _pred_name = mstar.classes_name[predict]
    _fp_dir = os.path.join(path, 'fp', _label_name, _pred_name)
    if not os.path.exists(_fp_dir):
        os.makedirs(_fp_dir, exist_ok=True)

    fig = plt.figure(figsize=(8, 4))
    fig.suptitle(f'{_label_name} : {_pred_name}')
    fig.add_subplot(1, 2, 1)
    plt.imshow(image[0, 0, :, :], cmap='gray')
    plt.axis('off')
    fig.add_subplot(1, 2, 2)
    plt.imshow(image[0, 1, :, :], cmap='gray')
    plt.axis('off')
    plt.savefig(os.path.join(_fp_dir, f'{index}.png'), bbox_inches='tight')

    plt.close('all')


@torch.no_grad()
def run(batch_size, num_classes, model_path=None):
    # Path to save the evaluation result
    result_path = os.path.dirname(os.path.dirname(model_path))
    model_name = os.path.basename(os.path.dirname(model_path))

    # Data set
    test_set = load_data(FLAGS.data_path, is_train=False, batch_size=batch_size)

    # Model
    m = model.Model(classes=num_classes)
    m.load(model_path)

    num_data = 0
    corrects = 0
    # Test loop
    confusion_matrix = np.zeros((num_classes, num_classes))

    m.net.eval()
    for i, data in enumerate(tqdm(test_set)):
        images, labels = data

        predictions = m.inference(images)

        _, predictions = torch.max(predictions.data, 1)
        labels = labels.type(torch.LongTensor)

        confusion_matrix[labels[:, 0], predictions[:]] += 1

        num_data += labels.size(0)
        corrects += (predictions == labels[:, 0].to(m.device)).sum().item()

        # save the false positive class image
        if not predictions == labels[:, 0].to(m.device):
            write_fp_image(result_path, labels[:, 0], predictions[:], i, images)

    # Print out the evaluation accuracy
    accuracy = 100 * corrects / num_data
    logging.info(f'Accuracy : {accuracy}%')

    logging.info('Confusion Matrix:')
    logging.info(f'Classes: {[mstar.classes_name[i] for i in range(10)]}')
    logging.info(f'\n{confusion_matrix}')

    # save the evaluation log
    write_evaluation_log(os.path.join(result_path, f'eval-{model_name}.csv'), confusion_matrix)


def main(_):
    # Flags
    experiments_path = FLAGS.experiments_path
    config_name = FLAGS.config_name
    epoch = FLAGS.epoch

    # load configurations
    config = utils.load_config(os.path.join(experiments_path, config_name))
    model_name = config['model_name']
    num_classes = config['num_classes']
    batch_size = 1  # config['batch_size']

    # initialize experimental path
    if not os.path.exists(os.path.join(experiments_path, model_name)):
        os.makedirs(os.path.join(experiments_path, model_name), exist_ok=True)

    run(batch_size, num_classes, os.path.join(experiments_path, model_name, f'model-{epoch:03d}.pth'))


if __name__ == '__main__':
    app.run(main)
