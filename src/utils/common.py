import numpy as np
import torch

import random
import json
import os

project_root = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


def set_random_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    np.random.seed(random_seed)
    random.seed(random_seed)


def load_config(path):
    with open(path, mode='r', encoding='utf-8') as f:
        return json.load(f)
