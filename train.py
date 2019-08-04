# -*- coding: utf-8 -*-

"""
Train
Author :
    Yuki Kumon
Last Update :
    2019-08-04
"""


from torch.utils.data import Dataset
from torchvision import transforms
import torch.nn as nn  # ネットワーク構築用
import torch.optim as optim  # 最適化関数
import torch.nn.functional as F  # ネットワーク用の様々な関数
import torch.utils.data  # データセット読み込み関連

import numpy as np
from PIL import Image

# from torchsummary import summary
from tensorboardX import SummaryWriter

# import matplotlib.pyplot as plt

from misc.attr_dict import AttributeDict
from absl import app, flags, logging


flags.DEFINE_string('cifar_path', './data/cifar-10-batches-py', 'cifar 10 dataset path')
